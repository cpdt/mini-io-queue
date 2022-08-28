//! Synchronous reader/writer queue for generic items or byte arrays.
//!
//! Each queue has a [`Reader`] and [`Writer`] part. Data can be copied into the writer's buffer and
//! sent to the reader allocations, allowing nonblocking communication across threads.
//!
//! Reading and writing with the queue does not require any allocation, with the downside that the
//! queue has a fixed capacity on creation.
//!
//! Unlike [`nonblocking`], this queue blocks to wait for data on the reader end, or wait for space
//! on the writer end. For [`u8`] storage, this means the queue can be used as a [`Read`] or
//! [`Write`].
//!
//! If you are using an async runtime, you are probably more interested in the [`asyncio`] queue,
//! which does not block.
//!
//! # Example
//! ```
//! use mini_io_queue::blocking::queue;
//!
//! let (mut reader, mut writer) = queue(8);
//!
//! let write_thread = std::thread::spawn(move || {
//!     for i in 0..16 {
//!         writer.write(&[i]);
//!     }
//! });
//!
//! let read_thread = std::thread::spawn(move || {
//!     for i in 0..16 {
//!         let mut buf = [0];
//!         reader.read_exact(&mut buf).unwrap();
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
//! [`nonblocking`]: crate::nonblocking
//! [`Read`]: std::io::Read
//! [`Write`]: std::io::Write
//! [`asyncio`]: crate::asyncio

use crate::storage::Storage;
use crate::{Region, RegionMut, Ring};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::{error, fmt, io};

/// Creates a queue that is backed by a specific storage. The queue will use the storage's entire
/// capacity, and will be initialized with an empty read buffer and a full write buffer.
///
/// Note that the reader and writer will only implement [`Send`] and [`Sync`] if the storage also
/// does.
///
/// # Example
/// ```
/// use mini_io_queue::blocking::queue_from;
/// use mini_io_queue::storage::HeapBuffer;
///
/// let buffer = HeapBuffer::<u8>::new(100);
/// let (reader, writer) = queue_from(buffer);
/// ```
///
/// [`Send`]: std::marker::Send
/// [`Sync`] std::marker::Sync
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
/// use mini_io_queue::blocking::queue_from_parts;
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

        data_available_cond: Condvar::new(),
        space_available_cond: Condvar::new(),
    });

    let reader = Reader {
        state: state.clone(),
        data_available_mutex: Mutex::new(()),
    };
    let writer = Writer {
        state,
        space_available_mutex: Mutex::new(()),
    };

    (reader, writer)
}

#[cfg(feature = "heap-buffer")]
mod heap_constructors {
    use crate::blocking::{queue_from_parts, Reader, Writer};
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
    /// use mini_io_queue::blocking::queue;
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

#[derive(Debug)]
struct State<S> {
    ring: Ring,
    storage: S,

    is_reader_open: AtomicBool,
    is_writer_open: AtomicBool,

    data_available_cond: Condvar,
    space_available_cond: Condvar,
}

/// An error indicating why a writer failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteError {
    /// Writing failed because the reader was closed, preventing the read buffer from emptying.
    ReaderClosed,
}

/// An error indicating why reading failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadExactError {
    /// Reading failed because the writer was closed, meaning no more data will become available.
    WriterClosed,
}

/// Receives items from the queue.
///
/// Values sent by the writer will be added to the end of the reader's buffer, and capacity can be
/// sent back to the writer from the start of the reader's buffer to allow it to write more data.
#[derive(Debug)]
pub struct Reader<S> {
    state: Arc<State<S>>,
    data_available_mutex: Mutex<()>,
}

/// Adds items to the queue.
///
/// Values sent by the writer will be added to the end of the reader's buffer, and capacity can be
/// sent back to the writer from the start of the reader's buffer to allow it to write more data.
#[derive(Debug)]
pub struct Writer<S> {
    state: Arc<State<S>>,
    space_available_mutex: Mutex<()>,
}

impl<S> State<S> {
    fn close_reader(&self) {
        let was_open = self.is_reader_open.swap(false, Ordering::AcqRel);
        if was_open {
            self.space_available_cond.notify_all();
        }
    }

    fn close_writer(&self) {
        let was_open = self.is_writer_open.swap(false, Ordering::AcqRel);
        if was_open {
            self.data_available_cond.notify_all();
        }
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
    /// If this is true it is guaranteed that the next call to [`fill_buf`] will return a non-empty
    /// slice, unless [`consume`] is called first.
    ///
    /// Keep in mind that when using a reader and writer on separate threads, a reader that has no
    /// data can receive data at any time - even between calls to `has_data` and other functions.
    ///
    /// [`fill_buf`]: Reader::fill_buf
    /// [`consume`]: Reader::consume
    #[inline]
    pub fn has_data(&self) -> bool {
        let (r1, r2) = self.state.ring.left_ranges();
        !r1.is_empty() || !r2.is_empty()
    }

    /// Returns if the buffer is full, i.e all space is allocated to the reader and any write
    /// operations will block.
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

    /// Attempt to read from the reader's buffer, blocking to wait for more data if it is empty.
    ///
    /// This function is a lower-level call. It needs to be paired with the [`consume`] method to
    /// function properly. When calling this method, none of the contents will be "read" in the
    /// sense that later calling `fill_buf` may return the same contents. As such, [`consume`] must
    /// be called with the number of bytes that are consumed from this buffer to ensure that the
    /// items are never returned twice.
    ///
    /// An empty buffer returned indicates that all data has been read and the writer has closed.
    ///
    /// [`consume`]: Reader::consume
    pub fn fill_buf<T>(&mut self) -> Region<T>
    where
        S: Storage<T>,
    {
        if self.has_data() {
            return self.buf();
        }

        // If the writer is closed, we've now read everything we could.
        if !self.is_writer_open() {
            // Empty slice indicates the writer closed.
            return Default::default();
        }

        // If no data is available, park and ask the writer to unpark us when it writes something.
        let mut lock = self.data_available_mutex.lock().unwrap();
        loop {
            lock = self.state.data_available_cond.wait(lock).unwrap();

            if self.has_data() {
                return self.buf();
            }

            // If the writer is closed, we've now read everything we could.
            if !self.is_writer_open() {
                // Empty slice indicates the writer closed.
                return Default::default();
            }
        }
    }

    /// Marks items at the start of the reader buffer as consumed, removing them from the slice
    /// returned by [`fill_buf`] and adding their capacity to the end of the writer's buffer.
    /// Since queues have a fixed underlying length, calling this is required to allow the transfer
    /// of more data.
    ///
    /// # Panics
    /// This function will panic if `amt` is larger than the reader's available data buffer.
    ///
    /// [`fill_buf`]: Reader::fill_buf
    pub fn consume(&mut self, amt: usize) {
        self.state.ring.advance_left(amt);

        // Unpark the writer if it was waiting for space.
        self.state.space_available_cond.notify_all();
    }

    /// Pulls some items from this queue into the specified buffer, returning how many items were
    /// read.
    ///
    /// This method will complete immediately if at least one item is available to read, otherwise
    /// it will block until some are available.
    ///
    /// # Return
    /// It is guaranteed that the return value is `<= buf.len()`.
    ///
    /// A return value of `0` indicates one of these two scenarios:
    ///  1. The writer has closed and all items have been read.
    ///  2. The buffer specified had a length of 0.
    pub fn read<T>(&mut self, buf: &mut [T]) -> usize
    where
        S: Storage<T>,
        T: Clone,
    {
        let src_buf = self.fill_buf();

        if src_buf.is_empty() {
            // This indicates the writer has closed and all data has been read.
            return 0;
        }

        let len = src_buf.len().min(buf.len());
        src_buf.slice(..len).clone_to_slice(&mut buf[..len]);

        self.consume(len);
        len
    }

    /// Reads the exact number of items required to fill `buf`.
    ///
    /// If the writer closes before the buffer is completely filled, an error of the kind
    /// [`ReadExactError::WriterClosed`] will be returned.
    ///
    /// # Return
    /// If the return value is `Ok(n)`, it is guaranteed that `n == buf.len()`.
    pub fn read_exact<T>(&mut self, buf: &mut [T]) -> Result<usize, ReadExactError>
    where
        S: Storage<T>,
        T: Clone,
    {
        let len = buf.len();
        let src_buf = loop {
            let src_buf = self.fill_buf();

            if src_buf.len() >= len {
                break src_buf;
            }

            if !self.is_writer_open() {
                // The writer has closed, required data will never be ready.
                return Err(ReadExactError::WriterClosed);
            }
        };

        src_buf.slice(..len).clone_to_slice(buf);
        self.consume(len);

        Ok(len)
    }

    /// Close the reader, indicating to the writer that no more data will be read.
    ///
    /// Any in-progress writes or flushes on the writer will be interrupted, and any future
    /// operations will fail. Closing the reader multiple times has no effect.
    ///
    /// Dropping the reader object will also close it.
    #[inline]
    pub fn close(&mut self) {
        self.state.close_reader();
    }

    #[inline]
    fn buf<T>(&self) -> Region<T>
    where
        S: Storage<T>,
    {
        let (range_0, range_1) = self.state.ring.left_ranges();
        Region::new(
            self.state.storage.slice(range_0),
            self.state.storage.slice(range_1),
        )
    }
}

impl<S> io::Read for Reader<S>
where
    S: Storage<u8>,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let src_buf = self.fill_buf();

        let len = src_buf.len().min(buf.len());
        src_buf.slice(..len).copy_to_slice(&mut buf[..len]);

        self.consume(len);

        Ok(len)
    }
}

impl<S> io::BufRead for Reader<S>
where
    S: Storage<u8>,
{
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        Ok(self.fill_buf().contiguous())
    }

    fn consume(&mut self, amt: usize) {
        self.consume(amt);
    }
}

impl<S> Drop for Reader<S> {
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
    /// If this is true it is guaranteed that the next call to [`empty_buf`] will return a non-empty
    /// slice, unless [`feed`] is called first.
    ///
    /// Keep in mind that when using a reader and writer on separate threads, a writer that has no
    /// space can have more made available at any time - even between calls to `has_space` and other
    /// functions.
    ///
    /// [`empty_buf`]: Writer::empty_buf
    /// [`feed`]: Writer::feed
    #[inline]
    pub fn has_space(&self) -> bool {
        let (r0, r1) = self.state.ring.right_ranges();
        !r0.is_empty() || !r1.is_empty()
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
        let (r0, r1) = self.state.ring.left_ranges();
        r0.is_empty() && r1.is_empty()
    }

    fn get_flush_state(&self) -> Option<Result<(), WriteError>> {
        if self.is_flushed() {
            return Some(Ok(()));
        }
        if !self.is_reader_open() {
            return Some(Err(WriteError::ReaderClosed));
        }
        None
    }

    #[inline]
    fn buf<T>(&mut self) -> RegionMut<T>
    where
        S: Storage<T>,
    {
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

    /// Attempt to get the writable buffer, blocking to wait for more space if it is empty.
    ///
    /// This functions is a lower-level call. It needs to be paired with the [`feed`] method to
    /// function properly. When calling this method, none of the contents will be "written" in the
    /// sense that later calling `empty_buf` may return the same contents. As such, [`feed`] must be
    /// called with the number of items that have been written to the buffer to ensure that the
    /// items are never returned twice.
    ///
    /// An empty buffer returned indicates that the queue cannot be written to as the reader has
    /// closed.
    ///
    /// [`feed`]: Writer::feed
    pub fn empty_buf<T>(&mut self) -> RegionMut<T>
    where
        S: Storage<T>,
    {
        // If the reader is closed there is no point in writing anything, even if space is
        // available.
        if !self.is_reader_open() {
            // Empty slice indicates the reader closed.
            return Default::default();
        }
        if self.has_space() {
            return self.buf();
        }

        // If no space is available, park and ask the reader to unpark us when it writes something.
        {
            let mut lock = self.space_available_mutex.lock().unwrap();
            loop {
                lock = self.state.space_available_cond.wait(lock).unwrap();

                if !self.is_reader_open() {
                    // Empty slice indicates the reader closed.
                    return Default::default();
                }
                if self.has_space() {
                    break;
                }
            }
        }

        self.buf()
    }

    /// Marks items at the start of the writer buffer as ready to be read, removing them from the
    /// slice returned by [`empty_buf`] and making them available in the reader's buffer.
    ///
    /// # Panics
    /// This function will panic if `amt` is larger than the writer's available space buffer.
    ///
    /// [`empty_buf`]: Writer::empty_buf
    pub fn feed(&mut self, len: usize) {
        self.state.ring.advance_right(len);

        // Unpark the reader if it was waiting for data.
        self.state.data_available_cond.notify_all();
    }

    /// Writes some items from a buffer into this queue, returning how many items were written.
    ///
    /// This function will attempt to write the entire contents of `buf`, but the entire write may
    /// not succeed if not enough space is available.
    ///
    /// # Return
    /// It is guaranteed that the return value is `<= buf.len()`.
    ///
    /// A return value of `0` indicates one of these two scenarios:
    ///  1. The reader has closed.
    ///  2. The buffer specified had a length of 0.
    pub fn write<T>(&mut self, buf: &[T]) -> usize
    where
        S: Storage<T>,
        T: Clone,
    {
        let mut dest_buf = self.empty_buf();

        if dest_buf.is_empty() {
            // This indicates the reader has closed.
            return 0;
        }

        let len = dest_buf.len().min(buf.len());
        dest_buf.slice_mut(..len).clone_from_slice(&buf[..len]);

        self.feed(len);
        len
    }

    /// Attempts to write all items in a buffer into this queue.
    ///
    /// If the reader closes before all items are written, an error of the kind
    /// [`WriteError::ReaderClosed`] will be returned.
    ///
    /// # Return
    /// If the return value is `Ok(n)`, it is guaranteed that `n == buf.len()`.
    pub fn write_all<T>(&mut self, buf: &[T]) -> Result<usize, WriteError>
    where
        S: Storage<T>,
        T: Clone,
    {
        let len = buf.len();
        let mut dest_buf = loop {
            let dest_buf = self.empty_buf();

            if dest_buf.is_empty() {
                // The reader has closed.
                return Err(WriteError::ReaderClosed);
            }

            if dest_buf.len() >= len {
                break dest_buf;
            }
        };

        dest_buf.slice_mut(..len).clone_from_slice(buf);
        self.feed(len);

        Ok(len)
    }

    /// Flush the buffer, ensuring that any items waiting to be read are consumed by the reader.
    ///
    /// If the reader is closed, returns `FlushError::ReaderClosed`. If blocking cannot be completed
    /// immediately, this method blocks until the reader closed or the buffer is flushed.
    pub fn flush(&mut self) -> Result<(), WriteError> {
        if let Some(flush_state) = self.get_flush_state() {
            return flush_state;
        }

        // Wait for more data to be read.
        let mut lock = self.space_available_mutex.lock().unwrap();
        loop {
            lock = self.state.space_available_cond.wait(lock).unwrap();

            if let Some(flush_state) = self.get_flush_state() {
                return flush_state;
            }
        }
    }

    /// Close the writer, flushing any remaining data and indicating to the reader that no more data
    /// will be written.
    ///
    /// Any future read operations will fail. Closing the writer multiple times has no effect.
    pub fn close(&mut self) -> Result<(), WriteError> {
        self.flush()?;
        self.state.close_writer();
        Ok(())
    }
}

impl<S> io::Write for Writer<S>
where
    S: Storage<u8>,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut dest_buf = self.empty_buf();

        let len = dest_buf.len().min(buf.len());
        dest_buf.slice_mut(..len).copy_from_slice(&buf[..len]);

        self.feed(len);

        Ok(len)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush().map_err(Into::into)
    }
}

impl<S> Drop for Writer<S> {
    #[inline]
    fn drop(&mut self) {
        self.state.close_writer();
    }
}

impl error::Error for WriteError {}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteError::ReaderClosed => write!(f, "reader closed"),
        }
    }
}

impl From<WriteError> for io::Error {
    fn from(err: WriteError) -> Self {
        match err {
            WriteError::ReaderClosed => io::ErrorKind::UnexpectedEof.into(),
        }
    }
}

impl error::Error for ReadExactError {}

impl fmt::Display for ReadExactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadExactError::WriterClosed => write!(f, "writer closed"),
        }
    }
}

impl From<ReadExactError> for io::Error {
    fn from(err: ReadExactError) -> Self {
        match err {
            ReadExactError::WriterClosed => io::ErrorKind::UnexpectedEof.into(),
        }
    }
}
