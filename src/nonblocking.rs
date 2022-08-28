//! Non-blocking reader/writer queue for generic items or byte arrays.
//!
//! Each queue has a [`Reader`] and [`Writer`] part. Data can be copied into the writer's buffer
//! and sent to the reader without locks or allocation, allowing nonblocking communication across
//! threads.
//!
//! Reading and writing with the queue does not require any allocation, with the downside that the
//! queue has a fixed capacity on creation.
//!
//! `Reader` and `Writer` can implement [`Read`] and [`Write`] if the `std-io` feature is
//! enabled. These implementations will return [`WouldBlock`] errors instead of blocking.
//!
//! # Example
//! ```
//! use mini_io_queue::nonblocking::queue;
//!
//! let (mut reader, mut writer) = queue(8);
//!
//! let write_thread = std::thread::spawn(move || {
//!     for i in 0..16 {
//!         let buf = [i];
//!
//!         // spin until there is space to write
//!         loop {
//!             let write_len = writer.write(&buf);
//!             if write_len == 1 {
//!                 break;
//!             }
//!         }
//!     }
//! });
//!
//! let read_thread = std::thread::spawn(move || {
//!     for i in 0..16 {
//!         let mut buf = [0];
//!
//!         // spin until there is data to read
//!         loop {
//!             let read_len = reader.read(&mut buf);
//!             if read_len == 1 {
//!                 break;
//!             }
//!         }
//!
//!         assert_eq!(buf[0], i);
//!     }
//! });
//!
//! write_thread.join().unwrap();
//! read_thread.join().unwrap();
//! ```
//!
//! [`Reader`]: self::Reader
//! [`Writer`]: self::Writer
//! [`Read`]: std::io::Read
//! [`Write`]: std::io::Write
//! [`WouldBlock`]: std::io::ErrorKind::WouldBlock

use crate::storage::Storage;
use crate::{Region, RegionMut, Ring};
use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};

/// Creates a queue that is backed by a specific storage. The queue will use the storage's entire
/// capacity, and will be initialized with an empty read buffer and a full write buffer.
///
/// Note that the reader and writer will only implement [`Send`] and [`Sync`] if the storage also
/// does.
///
/// # Example
/// ```
/// use mini_io_queue::nonblocking::queue_from;
/// use mini_io_queue::storage::HeapBuffer;
///
/// let buffer = HeapBuffer::<u8>::new(100);
/// let (reader, writer) = queue_from(buffer);
/// ```
///
/// [`Send`]: std::marker::Send
/// [`Sync`]: std::marker::Sync
pub fn queue_from<T, S>(storage: S) -> (Reader<S>, Writer<S>)
where
    S: Storage<T>,
{
    let ring = Ring::new(storage.capacity());
    queue_from_parts(ring, storage)
}

/// Creates a queue from a separately allocated ring and storage. The queue will use the ring's
/// capacity, and be initialized with a read buffer from the ring's left region and a write buffer
/// from the ring's right region.
///
/// It is up to the user to ensure the storage has enough capacity for the ring. If the ring's
/// capacity is larger than the storage's length, the reader and writer may panic.
///
/// Note that the reader and writer will only implement [`Send`] and [`Sync`] if the storage also
/// does.
///
/// # Example
/// ```
/// use mini_io_queue::Ring;
/// use mini_io_queue::nonblocking::queue_from_parts;
/// use mini_io_queue::storage::{HeapBuffer, Storage};
///
/// // Create a queue with half of the underlying buffer in the read side.
/// let ring = Ring::new(10);
/// ring.advance_right(5);
///
/// let mut buffer = HeapBuffer::new(10);
/// buffer.slice_mut(0..5).copy_from_slice(&[1, 2, 3, 4, 5]);
///
/// let (reader, writer) = queue_from_parts(ring, buffer);
/// ```
///
/// [`Send`]: std::marker::Send
/// [`Sync`]: std::marker::Sync
pub fn queue_from_parts<S>(ring: Ring, storage: S) -> (Reader<S>, Writer<S>) {
    let state = Arc::new(State {
        ring,
        storage,

        is_reader_open: AtomicBool::new(true),
        is_writer_open: AtomicBool::new(true),
    });

    let reader = Reader {
        state: state.clone(),
    };
    let writer = Writer { state };

    (reader, writer)
}

#[cfg(feature = "heap-buffer")]
mod heap_constructors {
    use crate::nonblocking::{queue_from_parts, Reader, Writer};
    use crate::storage::HeapBuffer;
    use crate::Ring;

    /// Creates a queue with a specific capacity, allocating storage on the heap. The queue will
    /// be initialized with an empty read buffer and a full write buffer containing the element's
    /// default value.
    ///
    /// Note that the reader and writer will only implement [`Send`] and [`Sync`] if the element
    /// type also does.
    ///
    /// # Example
    /// ```
    /// use mini_io_queue::nonblocking::queue;
    ///
    /// let (reader, writer) = queue::<u8>(100);
    /// ```
    ///
    /// [`Send`]: std::marker::Send
    /// [`Sync`]: std::marker::Sync
    #[cfg_attr(docsrs, doc(cfg(feature = "heap-buffer")))]
    pub fn queue<T>(capacity: usize) -> (Reader<HeapBuffer<T>>, Writer<HeapBuffer<T>>)
    where
        T: Default,
    {
        let ring = Ring::new(capacity);
        let buffer = HeapBuffer::new(capacity);

        queue_from_parts(ring, buffer)
    }
}

#[cfg(feature = "heap-buffer")]
pub use self::heap_constructors::*;

struct State<S> {
    ring: Ring,
    storage: S,

    is_reader_open: AtomicBool,
    is_writer_open: AtomicBool,
}

/// Receives items from the queue.
///
/// Values sent by the writer will be added to the end of the reader's buffer, and capacity can be
/// sent back to the writer from the start of the reader's buffer to allow it to write more data.
pub struct Reader<S> {
    state: Arc<State<S>>,
}

/// Adds items to the queue.
///
/// Values sent by the writer will be added to the end of the reader's buffer, and capacity can be
/// sent back to the writer from the start of the reader's buffer to allow it to write more data.
pub struct Writer<S> {
    state: Arc<State<S>>,
}

impl<S> State<S> {
    fn close_reader(&self) {
        self.is_reader_open.store(false, Ordering::Release);
    }

    fn close_writer(&self) {
        self.is_writer_open.store(false, Ordering::Release);
    }
}

impl<S> Reader<S> {
    /// Returns if the corresponding writer is still open.
    ///
    /// If this is `false`, unread data will still be available to read but a well-behaved writer
    /// will not provide any new data.
    #[inline]
    pub fn is_writer_open(&self) -> bool {
        self.state.is_writer_open.load(Ordering::Acquire)
    }

    /// Returns if data is available in the reader's buffer.
    ///
    /// If this is true it is guaranteed that the next call to [`buf`] will return a non-empty
    /// slice, unless [`consume`] is called first.
    ///
    /// Keep in mind that when using a reader and writer on separate threads, a reader that has no
    /// data can receive data at any time - even between calls to `has_data` and other functions.
    ///
    /// [`buf`]: Reader::buf
    /// [`consume`]: Reader::consume
    #[inline]
    pub fn has_data(&self) -> bool {
        let (r1, r2) = self.state.ring.left_ranges();
        !r1.is_empty() || !r2.is_empty()
    }

    /// Returns if the buffer is full, i.e all space is allocated to the reader and any write
    /// operations will fail.
    ///
    /// If this is true a reader can only resume the writer by calling [`consume`] to pass capacity
    /// to the writer.
    ///
    /// Keep in mind that when using a reader and writer on separate threads, a reader that is not
    /// full can become full at any time - even between calls to `is_full` and other functions.
    ///
    /// [`consume`]: Reader::consume
    #[inline]
    pub fn is_full(&self) -> bool {
        let (r1, r2) = self.state.ring.right_ranges();
        r1.is_empty() && r2.is_empty()
    }

    /// Gets the readable buffer.
    ///
    /// This function is a lower-level call. It needs to be paired with the [`consume`] method to
    /// function properly. When calling this method, none of the contents will be "read" in the
    /// sense that later calling `buf` may return the same contents. As such, [`consume`] must
    /// be called with the number of bytes that are consumed from this buffer to ensure that the
    /// items are never returned twice.
    ///
    /// An empty buffer returned indicates that no data is available to read.
    ///
    /// # Panics
    /// This function may panic if the underlying storage panics when trying to get a slice to the
    /// data. This may happen if queue was created with a ring that has a larger capacity than the
    /// storage.
    ///
    /// [`consume`]: Reader::consume
    #[inline]
    pub fn buf<T>(&self) -> Region<T>
    where
        S: Storage<T>,
    {
        let (range_0, range_1) = self.state.ring.left_ranges();
        Region::new(
            self.state.storage.slice(range_0),
            self.state.storage.slice(range_1),
        )
    }

    /// Marks items at the start of the reader buffer as consumed, removing them from the slice
    /// returned by [`buf`] and adding their capacity to the end of the writer's buffer. Since
    /// queues have a fixed underlying length, calling this is required to allow the transfer of
    /// more data.
    ///
    /// # Panics
    /// This function will panic if `amt` is larger than the reader's available data buffer.
    ///
    /// [`buf`]: Reader::buf
    #[inline]
    pub fn consume(&mut self, amt: usize) {
        self.state.ring.advance_left(amt);
    }

    /// Pulls some items from this queue into the specified buffer, returning how many items were
    /// read.
    ///
    /// # Return
    /// It is guaranteed that the return value is `<= buf.len()`.
    ///
    /// A return value of `0` indicates one of these three scenarios:
    ///  1. No data was available to read.
    ///  2. The writer has closed and all items have been read.
    ///  3. The buffer specified had a length of 0.
    pub fn read<T>(&mut self, buf: &mut [T]) -> usize
    where
        S: Storage<T>,
        T: Clone,
    {
        let src_buf = self.buf();

        let len = src_buf.len().min(buf.len());
        src_buf.slice(..len).clone_to_slice(&mut buf[..len]);

        self.consume(len);
        len
    }

    /// Close the reader, indicating to the writer that no more data will be read.
    ///
    /// Any future write operations will fail. Closing the reader multiple times has no effect.
    ///
    /// Dropping the reader object will also close it.
    #[inline]
    pub fn close(&mut self) {
        self.state.close_reader();
    }
}

impl<S> Drop for Reader<S> {
    #[inline]
    fn drop(&mut self) {
        self.state.close_reader();
    }
}

impl<S> Writer<S> {
    /// Returns if the corresponding reader is still open.
    ///
    /// If this is `false`, any attempt to write or flush the object will fail.
    #[inline]
    pub fn is_reader_open(&self) -> bool {
        self.state.is_reader_open.load(Ordering::Acquire)
    }

    /// Returns if space is available in the writer's buffer.
    ///
    /// If this is true it is guaranteed that the next call to [`buf`] will return a non-empty
    /// slice, unless [`feed`] is called first.
    ///
    /// Keep in mind that when using a reader and writer on separate threads, a writer that has no
    /// space can have more made available at any time - even between calls to `has_space` and other
    /// functions.
    ///
    /// [`buf`]: Writer::buf
    /// [`feed`]: Writer::feed
    #[inline]
    pub fn has_space(&self) -> bool {
        let (r1, r2) = self.state.ring.right_ranges();
        !r1.is_empty() || !r2.is_empty()
    }

    /// Returns if the buffer is flushed, i.e there are no items to read and any read operations
    /// will stall.
    ///
    /// If this is true a writer can only resume the reader by calling [`feed`] to pass items to
    /// the reader.
    ///
    /// Keep in mind that when using a reader and writer on separate threads, a writer that is not
    /// flushed can become flushed at any time - even between calls to `is_flushed` and other
    /// functions.
    ///
    /// [`feed`]: Writer::feed
    #[inline]
    pub fn is_flushed(&self) -> bool {
        let (r1, r2) = self.state.ring.left_ranges();
        r1.is_empty() && r2.is_empty()
    }

    /// Gets the writable buffer.
    ///
    /// This functions is a lower-level call. It needs to be paired with the [`feed`] method to
    /// function properly. When calling this method, none of the contents will be "written" in the
    /// sense that later calling `buf` may return the same contents. As such, [`feed`] must be
    /// called with the number of items that have been written to the buffer to ensure that the
    /// items are never returned twice.
    ///
    /// An empty buffer returned indicates that no space is currently available, or the reader has
    /// closed.
    ///
    /// # Panics
    /// This function may panic if the underlying storage panics when trying to get a slice to the
    /// data. This may happen if queue was created with a ring that has a larger capacity than the
    /// storage.
    ///
    /// [`feed`]: Writer::feed
    pub fn buf<T>(&mut self) -> RegionMut<T>
    where
        S: Storage<T>,
    {
        // If the reader is closed there is no point in writing anything, even if space is
        // available.
        if !self.is_reader_open() {
            // Empty slice indicates the reader closed.
            return Default::default();
        }

        let (range_0, range_1) = self.state.ring.right_ranges();

        // `Ring` guarantees that a left region will only overlap a right region when this order
        // is followed:
        //  - Get the right region range
        //  - Advance the right region
        //  - Get the left region range
        // Given that the borrow checker prevents this here (`buf` and `consume` both take
        // &mut self), and assuming the Reader behaves correctly and does not hold references to the
        // left region's buffer while advancing it, there is no way to get another range that
        // overlaps this one.
        RegionMut::new(
            unsafe { self.state.storage.slice_mut_unchecked(range_0) },
            unsafe { self.state.storage.slice_mut_unchecked(range_1) },
        )
    }

    /// Mark items at the start of the writer buffer as ready to be read, removing them from the
    /// slice returned by [`buf`] and making them available in the reader's buffer.
    ///
    /// # Panics
    /// This function will panic if `amt` is larger than the writer's available space buffer.
    ///
    /// [`buf`]: Writer::buf
    #[inline]
    pub fn feed(&mut self, amt: usize) {
        self.state.ring.advance_right(amt);
    }

    /// Writes some items from a buffer into this queue, returning how many items were written.
    ///
    /// This function will attempt to write the entire contents of `buf`, but the entire write may
    /// not succeed if not enough space is available.
    ///
    /// # Return
    /// It is guaranteed that the return value is `<= buf.len()`.
    ///
    /// A return value of `0` indicates one of these three scenarios:
    ///  1. No space is available to write.
    ///  2. The reader has closed.
    ///  3. The buffer specified had a length of 0.
    pub fn write<T>(&mut self, buf: &[T]) -> usize
    where
        S: Storage<T>,
        T: Clone,
    {
        let mut dest_buf = self.buf();

        let len = dest_buf.len().min(buf.len());
        dest_buf.slice_mut(..len).clone_from_slice(&buf[..len]);

        self.feed(len);
        len
    }

    /// Close the writer, indicating to the reader that no more data will be written.
    ///
    /// Closing the writer multiple times has no effect.
    ///
    /// Dropping the writer object will also close it.
    #[inline]
    pub fn close(&mut self) {
        self.state.close_writer();
    }
}

impl<S> Drop for Writer<S> {
    #[inline]
    fn drop(&mut self) {
        self.state.close_writer();
    }
}

#[cfg(feature = "std-io")]
mod io_impls {
    use crate::nonblocking::{Reader, Writer};
    use crate::storage::Storage;
    use std::io::{BufRead, ErrorKind, Read, Result, Write};

    #[cfg_attr(docsrs, doc(cfg(feature = "std-io")))]
    impl<S> Read for Reader<S>
    where
        S: Storage<u8>,
    {
        fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
            let src_buf = self.buf();

            if src_buf.is_empty() {
                return if self.is_writer_open() {
                    // No data to read, this is an error since blocking would be required.
                    Err(ErrorKind::WouldBlock.into())
                } else {
                    // Writer is closed and all data has been read, return 0 to indicate EOF.
                    Ok(0)
                };
            }

            let len = src_buf.len().min(buf.len());
            src_buf.slice(..len).copy_to_slice(&mut buf[..len]);

            self.consume(len);

            Ok(len)
        }
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "std-io")))]
    impl<S> BufRead for Reader<S>
    where
        S: Storage<u8>,
    {
        fn fill_buf(&mut self) -> Result<&[u8]> {
            let buf = self.buf().contiguous();

            if !buf.is_empty() {
                return Ok(buf);
            }

            if self.is_writer_open() {
                // No data to read, this is an error since blocking would be required.
                return Err(ErrorKind::WouldBlock.into());
            }

            // Writer is closed and all data has been read, return an empty slice to indicate EOF.
            Ok(Default::default())
        }

        fn consume(&mut self, amt: usize) {
            self.consume(amt);
        }
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "std-io")))]
    impl<S> Write for Writer<S>
    where
        S: Storage<u8>,
    {
        fn write(&mut self, buf: &[u8]) -> Result<usize> {
            let mut dest_buf = self.buf();

            if !dest_buf.is_empty() {
                let len = dest_buf.len().min(buf.len());
                dest_buf.slice_mut(..len).copy_from_slice(&buf[..len]);

                self.feed(len);

                return Ok(len);
            }

            if !self.is_reader_open() {
                // Return an empty slice to indicate EOF.
                return Ok(Default::default());
            }

            // No space to write, this is an error since blocking would be required.
            Err(ErrorKind::WouldBlock.into())
        }

        fn flush(&mut self) -> Result<()> {
            if self.is_flushed() {
                return Ok(());
            }
            if self.is_reader_open() {
                return Err(ErrorKind::WouldBlock.into());
            }
            Err(ErrorKind::UnexpectedEof.into())
        }
    }
}

#[cfg(feature = "std-io")]
pub use self::io_impls::*;
