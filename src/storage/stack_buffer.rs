use array_init::array_init;
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::fmt;

/// Backing buffer for a queue, allocated on the stack.
///
/// All queues require a backing buffer which implements the [`Storage`] trait. This buffer stores
/// values in a contiguous array allocated on the stack. Compare this to [`HeapBuffer`] which uses
/// a contiguous array on the heap. Note that both the [`asyncio`] and [`nonblocking`] queues move
/// their buffer to the heap anyway.
///
/// # Example
///
/// ```
/// use mini_io_queue::nonblocking;
/// use mini_io_queue::storage::StackBuffer;
///
/// let buffer: StackBuffer<u8, 100> = StackBuffer::default();
/// let (reader, writer) = nonblocking::queue_from(buffer);
/// ```
///
/// [`Storage`]: crate::storage::Storage
/// [`HeapBuffer`]: crate::storage::HeapBuffer
/// [`asyncio`]: crate::asyncio
/// [`nonblocking`]: crate::nonblocking
#[cfg_attr(docsrs, doc(cfg(feature = "stack-buffer")))]
pub struct StackBuffer<T, const N: usize>([UnsafeCell<T>; N]);

unsafe impl<T, const N: usize> Send for StackBuffer<T, N> where T: Send {}
unsafe impl<T, const N: usize> Sync for StackBuffer<T, N> where T: Sync {}

impl<T, const N: usize> Default for StackBuffer<T, N>
where
    T: Default,
{
    fn default() -> Self {
        StackBuffer(array_init(|_| Default::default()))
    }
}

impl<T, const N: usize> AsRef<[UnsafeCell<T>]> for StackBuffer<T, N> {
    fn as_ref(&self) -> &[UnsafeCell<T>] {
        &self.0
    }
}

impl<T, const N: usize> From<[T; N]> for StackBuffer<T, N> {
    fn from(arr: [T; N]) -> Self {
        let arr = MaybeUninit::new(arr);
        let unsafe_cell_arr_ptr = arr.as_ptr() as *const [UnsafeCell<T>; N];

        // Safety: UnsafeCell is repr(transparent), so has the same alignment as T.
        let unsafe_cell_arr = unsafe { core::ptr::read(unsafe_cell_arr_ptr) };
        StackBuffer(unsafe_cell_arr)
    }
}

impl<T, const N: usize> fmt::Debug for StackBuffer<T, N> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("StackBuffer")
            .finish()
    }
}
