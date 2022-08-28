//! Async reader/writer queue for generic items or byte arrays.
//!
//! Each queue has a [`Reader`] and [`Writer`] part. Data can be copied into the writer's buffer and
//! sent to the reader without locks or allocation, allowing nonblocking communication across
//! threads.
//!
//! Reading and writing with the queue does not require any allocation, with the downside that the
//! queue has a fixed capacity on creation.
//!
//! Unlike [`nonblocking`], this queue allows asynchronously waiting for data on the reader end,
//! or waiting for space on the writer end. For [`u8`] storage, this means the queue can be used
//! as a [`futures::AsyncRead`] and [`futures::AsyncWrite`] if the `std-io` feature is enabled.
//!
//! If you are not using an async runtime, you are probably more interested in the [`blocking`]
//! queue, which blocks instead of waiting asynchronously.
//!
//! # Example
//! ```
//! use futures::join;
//! use futures::executor::block_on;
//! use mini_io_queue::asyncio::queue;
//!
//! let (mut reader, mut writer) = queue(8);
//!
//! let write_loop = async {
//!     for i in 0..16 {
//!         writer.write(&[i]).await;
//!     }
//! };
//!
//! let read_loop = async {
//!     for i in 0..16 {
//!         let mut buf = [0];
//!         reader.read_exact(&mut buf).await.unwrap();
//!
//!         assert_eq!(buf[0], i);
//!     }
//! };
//!
//! block_on(async { join!(write_loop, read_loop) });
//! ```
//!
//! [`Reader`]: self::Reader
//! [`Writer`]: self::Writer
//! [`nonblocking`]: crate::nonblocking
//! [`blocking`]: crate::blocking
//! [`futures::AsyncRead`]: futures::AsyncRead
//! [`futures::AsyncWrite`]: futures::AsyncWrite

use crate::storage::Storage;
use crate::{Region, RegionMut, Ring};
use alloc::sync::Arc;
use core::fmt;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Context, Poll};
use futures::task::AtomicWaker;

/// Creates a queue that is backed by a specific storage. The queue will use the storage's entire
/// capacity, and will be initialized with an empty read buffer and a full write buffer.
///
/// Note that the reader and writer will only implement [`Send`] and [`Sync`] if the storage also
/// does.
///
/// # Example
/// ```
/// use mini_io_queue::asyncio::queue_from;
/// use mini_io_queue::storage::HeapBuffer;
///
/// let buffer: HeapBuffer<u8> = HeapBuffer::new(100);
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
/// use mini_io_queue::asyncio::queue_from_parts;
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

        data_available_waker: AtomicWaker::new(),
        space_available_waker: AtomicWaker::new(),
    });

    let reader = Reader {
        state: state.clone(),
    };
    let writer = Writer { state };

    (reader, writer)
}

#[cfg(feature = "heap-buffer")]
mod heap_constructors {
    use crate::asyncio::{queue_from_parts, Reader, Writer};
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
    /// use mini_io_queue::asyncio::queue;
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

    data_available_waker: AtomicWaker,
    space_available_waker: AtomicWaker,
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
///
/// A reader will automatically [`close`] itself when dropped.
///
/// [`close`]: Reader::close
#[derive(Debug)]
pub struct Reader<S> {
    state: Arc<State<S>>,
}

/// Adds items to the queue.
///
/// Values sent by the writer will be added to the end of the reader's buffer, and capacity can be
/// sent back to the writer from the start of the reader's buffer to allow it to write more data.
///
/// A writer will automatically close itself when dropped.
#[derive(Debug)]
pub struct Writer<S> {
    state: Arc<State<S>>,
}

impl<S> State<S> {
    fn close_reader(&self) {
        let was_open = self.is_reader_open.swap(false, Ordering::AcqRel);
        if was_open {
            self.space_available_waker.wake();
        }
    }

    fn close_writer(&self) {
        let was_open = self.is_writer_open.swap(false, Ordering::AcqRel);
        if was_open {
            self.data_available_waker.wake();
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
    /// If this is true it is guaranteed that the next call to [`poll_fill_buf`] will return a
    /// non-empty slice, unless [`consume`] is called first.
    ///
    /// Keep in mind that when using a reader and writer on separate threads, a reader that has no
    /// data can receive data at any time - even between calls to `has_data` and other functions.
    ///
    /// [`poll_fill_buf`]: Reader::poll_fill_buf
    /// [`consume`]: Reader::consume
    #[inline]
    pub fn has_data(&self) -> bool {
        let (r0, r1) = self.state.ring.left_ranges();
        !r0.is_empty() || !r1.is_empty()
    }

    /// Returns if the buffer is full, i.e all space is allocated to the reader and any write
    /// operations will stall.
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
        let (r0, r1) = self.state.ring.right_ranges();
        r0.is_empty() && r1.is_empty()
    }

    /// Attempt to read from the reader's buffer, waiting for more data if it is empty.
    ///
    /// On success, returns `Poll::Ready(Ok(buf))`.
    ///
    /// If no data is available for reading, the method returns `Poll::Pending` and arranges for
    /// the current task to receive a notification when the writer provides data or is closed.
    ///
    /// This function is a lower-level call. It needs to be paired with the [`consume`] method to
    /// function properly. When calling this method, none of the contents will be "read" in the
    /// sense that later calling `poll_fill_buf` may return the same contents. As such,
    /// [`consume`] must be called with the number of items that are consumed from this buffer to
    /// ensure that the items are never returned twice.
    ///
    /// An empty buffer returned indicates that all data has been read and the writer has closed.
    ///
    /// [`consume`]: Reader::consume
    pub fn poll_fill_buf<T>(&mut self, cx: &mut Context<'_>) -> Poll<Region<T>>
    where
        S: Storage<T>,
    {
        if self.has_data() {
            return Poll::Ready(self.buf());
        }

        // If no data is available, ask the writer to wake us when it writes something.
        self.state.data_available_waker.register(cx.waker());

        // Check if data appeared between the last check and the waker being set.
        if self.has_data() {
            // Data became available, remove the waker to avoid unnecessarily waking this task.
            self.state.data_available_waker.take();

            return Poll::Ready(self.buf());
        }

        // If the writer is closed, we've now read everything we could.
        // This must be checked after registering a waker to ensure we will see new
        // `is_writer_closed` values if the writer closed while registering the waker.
        if !self.is_writer_open() {
            // Remove the waker to avoid unnecessarily waking this task now it's done.
            self.state.data_available_waker.take();

            // Empty slice indicates the writer closed.
            return Poll::Ready(Default::default());
        }

        // Still empty, park until writer indicates data available.
        Poll::Pending
    }

    /// Marks items at the start of the reader buffer as consumed, removing them from the slice
    /// returned by [`poll_fill_buf`] and adding their capacity to the end of the writer's buffer.
    /// Since queues have a fixed underlying length, calling this is required to allow the transfer
    /// of more data.
    ///
    /// # Panics
    /// This function will panic if `amt` is larger than the reader's available data buffer.
    ///
    /// [`poll_fill_buf`]: Reader::poll_fill_buf
    pub fn consume(&mut self, amt: usize) {
        self.state.ring.advance_left(amt);

        // Wake the writer if it was waiting for space.
        self.state.space_available_waker.wake();
    }

    /// Pulls some items from this queue into the specified buffer, returning how many items were
    /// read.
    ///
    /// This method will complete immediately if at least one item is available to be read.
    ///
    /// # Return
    /// It is guaranteed that the return value is `<= buf.len()`.
    ///
    /// A return value of `0` indicates one of these two scenarios:
    ///  1. The writer has closed and all items have been read.
    ///  2. The buffer specified had a length of 0.
    ///
    /// # Cancel safety
    /// This method is cancel safe. If you use it in a `select!` statement and some other branch
    /// completes first, then it is guaranteed that no data was read.
    pub async fn read<T>(&mut self, buf: &mut [T]) -> usize
    where
        S: Storage<T>,
        T: Clone,
    {
        Read { reader: self, buf }.await
    }

    /// Reads the exact number of items required to fill `buf`.
    ///
    /// If the writer closes before the buffer is completely filled, an error of the kind
    /// [`ReadExactError::WriterClosed`] will be returned.
    ///
    /// # Return
    /// If the return value is `Ok(n)`, it is guaranteed that `n == buf.len()`.
    ///
    /// # Cancel safety
    /// This method is cancel safe. If you use it in a `select!` statement and some other branch
    /// completes first, then it is guaranteed that no data was read.
    pub async fn read_exact<T>(&mut self, buf: &mut [T]) -> Result<usize, ReadExactError>
    where
        S: Storage<T>,
        T: Clone,
    {
        ReadExact { reader: self, buf }.await
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

impl<S> Drop for Reader<S> {
    /// Closes the reader.
    #[inline]
    fn drop(&mut self) {
        self.state.close_reader();
    }
}

struct Read<'a, T, S> {
    reader: &'a mut Reader<S>,
    buf: &'a mut [T],
}

impl<'a, T, S> Future for Read<'a, T, S>
where
    S: Storage<T>,
    T: Clone,
{
    type Output = usize;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = self.get_mut();

        let src_buf = match me.reader.poll_fill_buf(cx) {
            Poll::Ready(src_buf) => src_buf,
            Poll::Pending => return Poll::Pending,
        };

        if src_buf.is_empty() {
            // This indicates the writer has closed and all data has been read.
            return Poll::Ready(0);
        }

        let len = src_buf.len().min(me.buf.len());
        src_buf.slice(..len).clone_to_slice(&mut me.buf[..len]);

        me.reader.consume(len);
        Poll::Ready(len)
    }
}

struct ReadExact<'a, T, S> {
    reader: &'a mut Reader<S>,
    buf: &'a mut [T],
}

impl<'a, T, S> Future for ReadExact<'a, T, S>
where
    S: Storage<T>,
    T: Clone,
{
    type Output = Result<usize, ReadExactError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = self.get_mut();

        let src_buf = match me.reader.poll_fill_buf(cx) {
            Poll::Ready(src_buf) => src_buf,
            Poll::Pending => return Poll::Pending,
        };

        let len = me.buf.len();
        if src_buf.len() < len {
            return if me.reader.is_writer_open() {
                // Not enough data is ready to write to the buffer.
                Poll::Pending
            } else {
                // The writer has closed, required data will never be ready.
                Poll::Ready(Err(ReadExactError::WriterClosed))
            };
        }

        src_buf.slice(..len).clone_to_slice(me.buf);
        me.reader.consume(len);

        Poll::Ready(Ok(len))
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
    /// If this is true it is guaranteed that the next call to [`poll_empty_buf`] will return a
    /// non-empty slice, unless [`feed`] is called first.
    ///
    /// Keep in mind that when using a reader and writer on separate threads, a writer that has no
    /// space can have more made available at any time - even between calls to `has_space` and other
    /// functions.
    ///
    /// [`poll_empty_buf`]: Writer::poll_empty_buf
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

    /// Attempt to get the writable buffer, waiting for more space if it is empty.
    ///
    /// On success, returns `Poll::Ready(Ok(buf))`.
    ///
    /// If no space is available for writing, the method returns `Poll::Pending` and arranges for
    /// the current task to receive a notification when the reader provides space or is closed.
    ///
    /// This functions is a lower-level call. It needs to be paired with the [`feed`] method to
    /// function properly. When calling this method, none of the contents will be "written" in the
    /// sense that later calling `poll_empty_buf` may return the same contents. As such, [`feed`]
    /// must be called with the number of items that have been written to the buffer to ensure that
    /// the items are never returned twice.
    ///
    /// An empty buffer returned indicates that the queue cannot be written to as the reader has
    /// closed.
    ///
    /// [`feed`]: Writer::feed
    pub fn poll_empty_buf<T>(&mut self, cx: &mut Context<'_>) -> Poll<RegionMut<T>>
    where
        S: Storage<T>,
    {
        // If the reader is closed there is no point in writing anything, even if space
        // is available.
        if !self.is_reader_open() {
            // Empty slice indicates the reader closed.
            return Poll::Ready(Default::default());
        }

        if self.has_space() {
            return Poll::Ready(self.buf());
        }

        // If no space is available, ask the reader to wake us when it reads something.
        self.state.space_available_waker.register(cx.waker());

        // Check if the reader is closed again in case it was closed between the last check and the
        // waker being set.
        if !self.is_reader_open() {
            // Remove the waker to avoid unnecessarily waking this task now it's done.
            self.state.data_available_waker.take();

            // Empty slice indicates the reader closed.
            return Poll::Ready(Default::default());
        }

        // Check if space appeared between the last check and the waker being set.
        if self.has_space() {
            // Space became available remove the waker to avoid unnecessarily waking this task.
            self.state.data_available_waker.take();

            return Poll::Ready(self.buf());
        }

        // Still empty, park until reader indicates space available.
        Poll::Pending
    }

    /// Marks items at the start of the writer buffer as ready to be read, removing them from the
    /// slice returned by [`poll_empty_buf`] and making them available in the reader's buffer.
    ///
    /// # Panics
    /// This function will panic if `amt` is larger than the writer's available space buffer.
    ///
    /// [`poll_empty_buf`]: Writer::poll_empty_buf
    pub fn feed(&mut self, len: usize) {
        self.state.ring.advance_right(len);

        // Wake the reader if it was waiting for data
        self.state.data_available_waker.wake();
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
    ///
    /// # Cancel safety
    /// This method is cancel safe. If you use it in a `select!` statement and some other branch
    /// completes first, then it is guaranteed that no data was written.
    pub async fn write<T>(&mut self, buf: &[T]) -> usize
    where
        S: Storage<T>,
        T: Clone,
    {
        Write { writer: self, buf }.await
    }

    /// Attempts to write all items in a buffer into this queue.
    ///
    /// If the reader closes before all items are written, an error of the kind
    /// [`WriteError::ReaderClosed`] will be returned.
    ///
    /// # Return
    /// If the return value is `Ok(n)`, it is guaranteed that `n == buf.len()`.
    ///
    /// # Cancel safety
    /// This method is cancel safe. If you use it in a `select!` statement and some other branch
    /// completes first, then it is guaranteed that no data was written.
    pub async fn write_all<T>(&mut self, buf: &[T]) -> Result<usize, WriteError>
    where
        S: Storage<T>,
        T: Clone,
    {
        WriteAll { writer: self, buf }.await
    }

    /// Attempt to flush the buffer, ensuring that any items waiting to be read are consumed by the
    /// reader.
    ///
    /// On success, returns `Poll::Ready(Ok(()))`. If the reader is closed, returns
    /// `Poll::Ready(Err(FlushError::ReaderClosed))`.
    ///
    /// If flushing cannot immediately complete, this method returns `Poll::Pending` and arranges
    /// for the current task to receive a notification when the object can make progress towards
    /// flushing.
    pub fn poll_flush(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), WriteError>> {
        if let Some(flush_state) = self.get_flush_state() {
            return Poll::Ready(flush_state);
        }

        // Wait for more data to be read before checking if the data is flushed.
        self.state.space_available_waker.register(cx.waker());

        // Check the flush state again in case the reader read between the last check and the waker
        // being set.
        if let Some(flush_state) = self.get_flush_state() {
            // Flush is complete, remove the waker to avoid unnecessarily waking this task.
            self.state.space_available_waker.take();

            return Poll::Ready(flush_state);
        }

        Poll::Pending
    }

    /// Flushes the buffer, ensuring that any items waiting to be read are consumed by the reader.
    ///
    /// If the reader closes before the buffer is flushed, an error of the kind
    /// [`WriteError::ReaderClosed`] will be returned.
    pub async fn flush(&mut self) -> Result<(), WriteError> {
        Flush(self).await
    }

    /// Attempt to close the writer, flushing any remaining data and indicating to the reader that
    /// no more data will be written.
    ///
    /// On success, returns `Poll::Ready(Ok(()))`. Any future read operations will fail. Closing
    /// the writer multiple times has no effect.
    pub fn poll_close(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), WriteError>> {
        // Wait for data to be flushed.
        match self.poll_flush(cx) {
            Poll::Ready(Ok(())) => {}
            Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
            Poll::Pending => return Poll::Pending,
        }

        self.state.close_writer();
        Poll::Ready(Ok(()))
    }

    /// Closes the buffer, flushing any remaining data and indicating to the reader that no more
    /// data will be written.
    ///
    /// If the reader closes before the buffer is flushed, an error of the kind
    /// [`WriteError::ReaderClosed`] will be returned.
    pub async fn close(&mut self) -> Result<(), WriteError> {
        Flush(self).await
    }
}

impl<S> Drop for Writer<S> {
    /// Closes the writer without flushing.
    #[inline]
    fn drop(&mut self) {
        self.state.close_writer();
    }
}

struct Write<'a, T, S> {
    writer: &'a mut Writer<S>,
    buf: &'a [T],
}

impl<'a, T, S> Future for Write<'a, T, S>
where
    S: Storage<T>,
    T: Clone,
{
    type Output = usize;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = self.get_mut();

        let mut dest_buf = match me.writer.poll_empty_buf(cx) {
            Poll::Ready(dest_buf) => dest_buf,
            Poll::Pending => return Poll::Pending,
        };

        if dest_buf.is_empty() {
            // This indicates the reader has closed.
            return Poll::Ready(0);
        }

        let len = dest_buf.len().min(me.buf.len());
        dest_buf.slice_mut(..len).clone_from_slice(&me.buf[..len]);

        me.writer.feed(len);
        Poll::Ready(len)
    }
}

struct WriteAll<'a, T, S> {
    writer: &'a mut Writer<S>,
    buf: &'a [T],
}

impl<'a, T, S> Future for WriteAll<'a, T, S>
where
    S: Storage<T>,
    T: Clone,
{
    type Output = Result<usize, WriteError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let me = self.get_mut();

        let mut dest_buf = match me.writer.poll_empty_buf(cx) {
            Poll::Ready(dest_buf) => dest_buf,
            Poll::Pending => return Poll::Pending,
        };

        if dest_buf.is_empty() {
            // The reader has closed.
            return Poll::Ready(Err(WriteError::ReaderClosed));
        }

        let len = me.buf.len();
        if dest_buf.len() < len {
            // Not enough space is ready to write to the buffer.
            return Poll::Pending;
        }

        dest_buf.slice_mut(..len).clone_from_slice(me.buf);
        me.writer.feed(len);

        Poll::Ready(Ok(len))
    }
}

struct Flush<'a, S>(&'a mut Writer<S>);

impl<'a, S> Future for Flush<'a, S> {
    type Output = Result<(), WriteError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut().0.poll_flush(cx)
    }
}

struct Close<'a, S>(&'a mut Writer<S>);

impl<'a, S> Future for Close<'a, S> {
    type Output = Result<(), WriteError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut().0.poll_close(cx)
    }
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteError::ReaderClosed => write!(f, "reader closed"),
        }
    }
}

impl fmt::Display for ReadExactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadExactError::WriterClosed => write!(f, "writer closed"),
        }
    }
}

#[cfg(feature = "std")]
mod std_impls {
    use crate::asyncio::{ReadExactError, WriteError};
    use std::{error, io};

    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    impl error::Error for WriteError {}

    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    impl From<WriteError> for io::Error {
        fn from(err: WriteError) -> Self {
            match err {
                WriteError::ReaderClosed => io::ErrorKind::UnexpectedEof.into(),
            }
        }
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    impl error::Error for ReadExactError {}

    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    impl From<ReadExactError> for io::Error {
        fn from(err: ReadExactError) -> Self {
            match err {
                ReadExactError::WriterClosed => io::ErrorKind::UnexpectedEof.into(),
            }
        }
    }
}

#[cfg(feature = "std")]
pub use self::std_impls::*;

#[cfg(feature = "std-io")]
mod io_impls {
    use crate::asyncio::{Reader, Writer};
    use crate::storage::Storage;
    use core::pin::Pin;
    use core::task::{Context, Poll};
    use futures::{io, AsyncBufRead, AsyncRead, AsyncWrite};

    #[cfg_attr(docsrs, doc(cfg(feature = "std-io")))]
    impl<S> AsyncRead for Reader<S>
    where
        S: Storage<u8>,
    {
        fn poll_read(
            self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            let me = self.get_mut();

            let src_buf = match me.poll_fill_buf(cx) {
                Poll::Ready(src_buf) => src_buf,
                Poll::Pending => return Poll::Pending,
            };

            if src_buf.is_empty() {
                // This indicates the writer has closed and all data has been read.
                return Poll::Ready(Ok(0));
            }

            let len = src_buf.len().min(buf.len());
            src_buf.slice(..len).copy_to_slice(&mut buf[..len]);

            me.consume(len);

            Poll::Ready(Ok(len))
        }
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "std-io")))]
    impl<S> AsyncBufRead for Reader<S>
    where
        S: Storage<u8>,
    {
        fn poll_fill_buf(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<&[u8]>> {
            self.get_mut()
                .poll_fill_buf(cx)
                .map(|region| Ok(region.contiguous()))
        }

        fn consume(self: Pin<&mut Self>, amt: usize) {
            self.get_mut().consume(amt);
        }
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "std-io")))]
    impl<S> AsyncWrite for Writer<S>
    where
        S: Storage<u8>,
    {
        fn poll_write(
            self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            let me = self.get_mut();

            let mut dest_buf = match me.poll_empty_buf(cx) {
                Poll::Ready(dest_buf) => dest_buf,
                Poll::Pending => return Poll::Pending,
            };

            if dest_buf.is_empty() {
                // This indicates the reader has closed.
                return Poll::Ready(Ok(0));
            }

            let len = dest_buf.len().min(buf.len());
            dest_buf.slice_mut(..len).copy_from_slice(&buf[..len]);

            me.feed(len);

            Poll::Ready(Ok(len))
        }

        fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            self.get_mut().poll_flush(cx).map(|r| r.map_err(Into::into))
        }

        fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            self.get_mut().poll_close(cx).map(|r| r.map_err(Into::into))
        }
    }
}

#[cfg(feature = "std-io")]
pub use self::io_impls::*;
