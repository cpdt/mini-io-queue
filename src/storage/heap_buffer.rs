use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::fmt;

/// Backing buffer for a queue, allocated on the heap.
///
/// All queues require a backing buffer which implements the [`Storage`] trait. This buffer stores
/// values in a contiguous array allocated on the heap. Compare this to [`StackBuffer`] which uses
/// a contiguous array on the stack. Note that the [`asyncio`], [`blocking`] and [`nonblocking`]
/// queues move their buffer to the heap anyway.
///
/// # Example
///
/// ```
/// use mini_io_queue::nonblocking;
/// use mini_io_queue::storage::HeapBuffer;
///
/// let buffer: HeapBuffer<u8> = HeapBuffer::new(100);
/// let (reader, writer) = nonblocking::queue_from(buffer);
/// ```
///
/// [`Storage`]: crate::storage::Storage
/// [`StackBuffer`]: crate::storage::StackBuffer
/// [`asyncio`]: crate::asyncio
/// [`blocking`]: crate::blocking
/// [`nonblocking`]: crate::nonblocking
#[cfg_attr(docsrs, doc(cfg(feature = "heap-buffer")))]
pub struct HeapBuffer<T>(Box<[UnsafeCell<T>]>);

unsafe impl<T> Send for HeapBuffer<T> where T: Send {}
unsafe impl<T> Sync for HeapBuffer<T> where T: Sync {}

impl<T> HeapBuffer<T>
where
    T: Default,
{
    /// Creates a buffer with the provided capacity. All elements in the buffer will be initialized
    /// to the item's default value.
    pub fn new(capacity: usize) -> Self {
        let mut vec = Vec::new();
        vec.resize_with(capacity, Default::default);
        HeapBuffer(vec.into_boxed_slice())
    }
}

impl<T> AsRef<[UnsafeCell<T>]> for HeapBuffer<T> {
    fn as_ref(&self) -> &[UnsafeCell<T>] {
        self.0.as_ref()
    }
}

impl<T> From<Box<[T]>> for HeapBuffer<T> {
    fn from(boxed_slice: Box<[T]>) -> Self {
        // Safety: UnsafeCell is repr(transparent) so this is safe.
        // fixme: this might not be stable, ideally we should not rely on it.
        let unsafe_cell_box: Box<[UnsafeCell<T>]> = unsafe { core::mem::transmute(boxed_slice) };
        HeapBuffer(unsafe_cell_box)
    }
}

impl<T> fmt::Debug for HeapBuffer<T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("HeapBuffer")
            .finish()
    }
}
