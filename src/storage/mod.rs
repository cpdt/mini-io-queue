//! Types to define how data is stored in a queue.
//!
//! To access data in an [`asyncio`], [`blocking`] or [`nonblocking`] queue, the backing buffer must
//! implement the [`Storage`] trait. This trait acts as an _unsafe_ building block to access
//! multiple mutable regions in a buffer, which each queue builds on to provide safe abstractions.
//!
//! Two standard storage types are available out of the box:
//!  - [`HeapBuffer`] stores a contiguous array of items on the heap. This means the buffer can have
//!    a size chosen at runtime and is cheaper to move, with the possible downside of requiring an
//!    allocator.
//!  - [`StackBuffer`] stores a contiguous array of items on the stack. As a result, it has a fixed
//!    size at compile-time and can be expensive to move but is extremely cheap to create.
//!
//! Note: all queues currently move their buffers to the heap when constructed.
//!
//! [`asyncio`]: crate::asyncio
//! [`blocking`]: crate::blocking
//! [`nonblocking`]: crate::nonblocking
//!
//! # Examples
//!
//! Creating a queue from a heap-allocated buffer:
//!
//! ```
//! use mini_io_queue::blocking;
//! use mini_io_queue::storage::HeapBuffer;
//!
//! let buffer: HeapBuffer<u8> = HeapBuffer::new(100);
//! let (reader, writer) = blocking::queue_from(buffer);
//! ```
//!
//! Creating a queue from a stack-allocated buffer:
//!
//! ```
//! use mini_io_queue::blocking;
//! use mini_io_queue::storage::StackBuffer;
//!
//! let buffer: StackBuffer<u8, 100> = StackBuffer::default();
//! let (reader, writer) = blocking::queue_from(buffer);
//! ```

use core::cell::UnsafeCell;
use core::ops::Range;
use core::slice::{from_raw_parts, from_raw_parts_mut};

#[cfg(feature = "heap-buffer")]
mod heap_buffer;

#[cfg(feature = "stack-buffer")]
mod stack_buffer;

#[cfg(feature = "heap-buffer")]
pub use self::heap_buffer::*;

#[cfg(feature = "stack-buffer")]
pub use self::stack_buffer::*;

/// Storage for a contiguous array of items allowing mutable access to multiple ranges as long as
/// they don't overlap.
///
/// This is implemented for any type that implements `AsRef<[UnsafeCell<T>]>`. No matter the backing
/// store, elements must be wrapped in some kind of [cell], as this is the only safe way to get a
/// mutable reference to something through a non-mutable reference.
///
/// This is a low-level trait, and must be implemented carefully.
///
/// [cell]: core::cell
pub trait Storage<T> {
    /// Gets the capacity of the array, or in other words the upper bound for ranges passed into
    /// [`slice`] and [`slice_mut`] before they will panic.
    ///
    /// [`slice`]: Storage::slice
    /// [`slice_mut`]: Storage::slice_mut
    fn capacity(&self) -> usize;

    /// Gets a slice of elements in the range provided. The length of the slice will always match
    /// the length of the range.
    ///
    /// # Panics
    /// This function may panic if the range extends beyond the length of the array returned by
    /// [`capacity`].
    ///
    /// [`capacity`]: Storage::capacity
    fn slice(&self, range: Range<usize>) -> &[T];

    /// Gets a mutable slice of elements in the range provided. The length of the slice will always
    /// match the length of the range. This function is unchecked and unsafe because it does not
    /// ensure there are no other references overlapping with the range, which is against Rust's
    /// borrowing rules and is **very unsafe!**
    ///
    /// This function does ensure the range is valid, however.
    ///
    /// # Safety
    /// No other slices that overlap with the range must exist for the duration of this slice's
    /// lifetime.
    ///
    /// # Panics
    /// This function may panic if the range extends beyond the length of the array returned by
    /// [`capacity`].
    ///
    /// [`capacity`]: Storage::capacity
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice_mut_unchecked(&self, range: Range<usize>) -> &mut [T];

    /// Gets a mutable slice of elements in the range provided. The length of the slice will always
    /// match the length of the range.
    ///
    /// # Panics
    /// This function may panic if the range extends beyond the length of the array returned by
    /// [`capacity`].
    ///
    /// [`capacity`]: Storage::capacity
    #[inline]
    fn slice_mut(&mut self, range: Range<usize>) -> &mut [T] {
        // Safety: the borrow checker prevents overlapping slices existing.
        unsafe { self.slice_mut_unchecked(range) }
    }
}

impl<T, A: AsRef<[UnsafeCell<T>]>> Storage<T> for A {
    fn capacity(&self) -> usize {
        self.as_ref().len()
    }

    fn slice(&self, range: Range<usize>) -> &[T] {
        let slice = &self.as_ref()[range];

        let first_elem = match slice.first() {
            Some(val) => val,
            None => return Default::default(),
        };

        // Access the slice of UnsafeCells as a slice of inner values.
        let slice_ptr = first_elem.get();

        // Safety:
        //  - len is guaranteed to be in range since the slice is valid.
        //  - UnsafeCell is repr(transparent) so an UnsafeCell<T> slice has the same layout and
        //    alignment as a T.
        unsafe { from_raw_parts(slice_ptr, slice.len()) }
    }

    unsafe fn slice_mut_unchecked(&self, range: Range<usize>) -> &mut [T] {
        let slice = &self.as_ref()[range];

        let first_elem = match slice.first() {
            Some(val) => val,
            None => return Default::default(),
        };

        // Access the slice of UnsafeCells as a mutable slice of inner values.
        let slice_ptr = first_elem.get();

        // Safety:
        //  - len is guaranteed to be in range since the slice is valid.
        //  - UnsafeCell is repr(transparent) so an UnsafeCell<T> slice has the same layout and
        //    alignment as a T.
        //  - caller ensures no overlapping slices exist.
        from_raw_parts_mut(slice_ptr, slice.len())
    }
}
