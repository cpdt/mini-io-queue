use core::ops::Range;
use core::sync::atomic::{AtomicUsize, Ordering};
use crate::cache_padded::CachePadded;

/// A low-level atomic ring-buffer building block.
///
/// This type implements logic for managing the two non-overlapping regions of a ring buffer mapped
/// to linear storage. This is useful for implementing safe, higher-level queues such as
/// [`nonblocking`] and [`asyncio`].
///
/// `Ring` models two "regions", named the left and right region, of a conceptual fixed-length ring
/// array. Each region can advance itself, moving data at the start of the region to the end of the
/// other region. This allows data in each region to be read and written independently, then fed to
/// the other region without synchronization.
///
/// When a region is mapped to a linear array, which has a start and end index, it might end up in
/// two "chunks" where it overlaps the edge of the array. `Ring` provides access to the indices
/// of the first chunks in the left and right region, as well as logic to advance each region. It
/// does not contain the buffer itself.
///
/// [`nonblocking`]: crate::nonblocking
/// [`asyncio`]: crate::asyncio
#[derive(Debug, Default)]
pub struct Ring {
    left: CachePadded<AtomicUsize>,
    right: CachePadded<AtomicUsize>,
    capacity: usize,
}

impl Ring {
    /// Creates a ring with a specific capacity.
    ///
    /// The ring starts with an empty left region and a right range from index 0 to the capacity.
    pub fn new(capacity: usize) -> Self {
        Ring {
            left: CachePadded::new(AtomicUsize::new(0)),
            right: CachePadded::new(AtomicUsize::new(0)),
            capacity,
        }
    }

    /// Gets the capacity of the ring.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Gets the range of indices in both chunks of the left region. Both or one range can be empty.
    /// The ranges will never overlap.
    ///
    /// The function guarantees that the start index of both ranges will stay the same across calls,
    /// and the end index will only increase, until the range is invalidated.
    ///
    /// This function also guarantees that the returned ranges will never overlap with a range
    /// returned by [`right_ranges`], as long as the ranges are not invalidated.
    ///
    /// The ranges are invalidated by advancing the region by calling [`advance_left`] or
    /// [`advance_left_unchecked`]. Any ranges read before advancing are effectively meaningless
    /// after advancing. Invalidated ranges must not be used to slice an underlying array, or you
    /// may end up with overlapping left and right slices.
    ///
    /// [`advance_left`]: Ring::advance_left
    /// [`advance_left_unchecked`]: Ring::advance_left_unchecked
    /// [`right_ranges`]: Ring::right_ranges
    pub fn left_ranges(&self) -> (Range<usize>, Range<usize>) {
        let left = self.left.load(Ordering::Acquire);
        let right = self.right.load(Ordering::Acquire);

        let left_offset = left % self.capacity;
        let right_offset = right % self.capacity;

        debug_assert!(left_offset <= self.capacity);
        debug_assert!(right_offset <= self.capacity);

        // Left is empty if read == write
        if left == right {
            return (left_offset..left_offset, left_offset..left_offset);
        }

        if left_offset >= right_offset {
            //  V chunk 2      V chunk 1
            // [LLLRRRRRRRRRRRRLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL]
            //     ^-right     ^-left

            (left_offset..self.capacity, 0..right_offset)
        } else {
            //     V chunk 1
            // [RRRLLLLLLLLLLLLRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR]
            //     ^-left      ^-right

            (left_offset..right_offset, 0..0)
        }
    }

    /// Gets the total length of the left region. This will always match the combined lengths of the
    /// slices returned by [`left_ranges`]. This will always be less than or equal to the ring's
    /// capacity.
    ///
    /// [`left_ranges`]: Ring::left_ranges
    pub fn left_len(&self) -> usize {
        let left = self.left.load(Ordering::Acquire);
        let right = self.right.load(Ordering::Acquire);

        right.wrapping_sub(left) % (self.capacity * 2)
    }

    /// Advances the left region, conceptually moving `len` elements at the start of the left region
    /// to the end of the right region and shrinking the left region as a result.
    ///
    /// # Panics
    /// Panics if `len` is larger than the current size of the left region.
    pub fn advance_left(&self, len: usize) {
        assert!(
            len <= self.left_len(),
            "len was larger than left region length"
        );

        // Safety: max length is ensured by the assert.
        unsafe { self.advance_left_unchecked(len) }
    }

    /// Advances the left region, conceptually moving `len` elements at the start of the left region
    /// to the end of the right region and shrinking the left region as a result.
    ///
    /// # Safety
    /// `len` must be less than or equal to the length of the left region returned by [`left_len`].
    ///
    /// [`left_len`]: Ring::left_len
    #[inline]
    pub unsafe fn advance_left_unchecked(&self, len: usize) {
        self.left
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |left| {
                Some(left.wrapping_add(len) % (self.capacity * 2))
            })
            .unwrap();
    }

    /// Gets the range of indices in the both chunks of the right region. Both or one range can be
    /// empty. The ranges will never overlap.
    ///
    /// The function guarantees that the start index of both ranges will stay the same across calls,
    /// and the end index will only increase, until the range is invalidated.
    ///
    /// This function also guarantees that the returned rangse will never overlap with a range
    /// returned by [`left_ranges`], as long as the ranges are not invalidated.
    ///
    /// The ranges are invalidated by advancing the region by calling [`advance_right`] or
    /// [`advance_right_unchecked`]. Any ranges read before advancing are effectively meaningless
    /// after advancing. Invalidated ranges must not be used to slice an underlying array, or you
    /// may end up with overlapping left and right slices.
    ///
    /// [`advance_right`]: Ring::advance_right
    /// [`advance_right_unchecked`]: Ring::advance_right_unchecked
    /// [`left_ranges`]: Ring::left_ranges
    pub fn right_ranges(&self) -> (Range<usize>, Range<usize>) {
        let left = self.left.load(Ordering::Acquire);
        let right = self.right.load(Ordering::Acquire);

        let left_size = right.wrapping_sub(left) % (self.capacity * 2);

        let left_offset = left % self.capacity;
        let right_offset = right % self.capacity;

        debug_assert!(left_offset <= self.capacity);
        debug_assert!(right_offset <= self.capacity);

        // Right is empty if left_size == capacity
        if left_size == self.capacity {
            return (right_offset..right_offset, right_offset..right_offset);
        }

        if left_offset <= right_offset {
            //  V chunk 2      V chunk 1
            // [RRRLLLLLLLLLLLLRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR]
            //     ^-left      ^-right

            (right_offset..self.capacity, 0..left_offset)
        } else {
            //       V chunk 1
            // [LLLLLRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL]
            //       ^-right                               ^-left

            (right_offset..left_offset, 0..0)
        }
    }

    /// Gets the total length of the right region. This will always match the combined lengths of
    /// the slices returned by [`right_ranges`]. This will always be less than or equal to the
    /// ring's capacity.
    ///
    /// [`right_ranges`]: Ring::right_ranges
    pub fn right_len(&self) -> usize {
        self.capacity - self.left_len()
    }

    /// Advances the right region, conceptually moving `len` elements at the start of the right
    /// region to the end of the left region and shrinking the right region as a result.
    ///
    /// # Panics
    /// Panics if `len` is larger than the current size of the right region.
    pub fn advance_right(&self, len: usize) {
        assert!(
            len <= self.right_len(),
            "len was larger than right region length"
        );

        // Safety: max length is ensured by the assert.
        unsafe { self.advance_right_unchecked(len) }
    }

    /// Advances the right region, conceptually moving `len` elements at the start of the right
    /// region to the end of the left region and shrinking the right region as a result.
    ///
    /// # Safety
    /// `len` must be less than or equal to the length of the left region returned by [`right_len`].
    ///
    /// [`right_len`]: Ring::right_len
    #[inline]
    pub unsafe fn advance_right_unchecked(&self, len: usize) {
        self.right
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |right| {
                Some(right.wrapping_add(len) % (self.capacity * 2))
            })
            .unwrap();
    }
}
