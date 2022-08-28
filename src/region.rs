use core::ops::{Bound, Index, IndexMut, Range, RangeBounds};

/// A region of two separate slices that represent continuous data.
///
/// [`Ring`] implements a "ring-buffer", meaning the left and right regions can overlap the ends of
/// the underlying buffer, causing them to be split into two contiguous slices (chunks). `Region`
/// represents these two chunks, allowing their data to be accessed mostly like one continuous slice
/// without allocations or copies.
///
/// [`Ring`]: crate::Ring
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Region<'a, T>(&'a [T], &'a [T]);

impl<'a, T> Region<'a, T> {
    /// Creates a region with two slices.
    ///
    /// The slices should represent continuous data, where `slice_0` goes before `slice_1`.
    #[inline]
    pub fn new(slice_0: &'a [T], slice_1: &'a [T]) -> Self {
        Region(slice_0, slice_1)
    }

    /// Returns `true` if the region has a length of 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty() && self.1.is_empty()
    }

    /// Returns the number of elements in the region.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len() + self.1.len()
    }

    /// Returns an iterator over the region.
    ///
    /// The iterator yields all items from start to end.
    pub fn iter(&self) -> impl Iterator<Item = &'a T> + 'a {
        [self.0.iter(), self.1.iter()].into_iter().flatten()
    }

    /// Returns the first contiguous slice in the region.
    ///
    /// The slice will not necessarily contain all data in the region. If the region is not empty,
    /// the slice is guaranteed to contain some data.
    #[inline]
    pub fn contiguous(&self) -> &'a [T] {
        if self.0.is_empty() {
            self.1
        } else {
            self.0
        }
    }

    /// Slices the region, returning a new region containing a part of the original.
    ///
    /// # Panics
    /// Panics if the `start` or `end` of the range is `>= region.len()`.
    pub fn slice<R>(&self, range: R) -> Self
    where
        R: RangeBounds<usize>,
    {
        let (range_0, range_1) = self.slice_ranges(range);

        let slice_0 = &self.0[range_0];
        let slice_1 = &self.1[range_1];

        Region(slice_0, slice_1)
    }

    fn slice_ranges<R>(&self, range: R) -> (Range<usize>, Range<usize>)
    where
        R: RangeBounds<usize>,
    {
        let start_inclusive = match range.start_bound() {
            Bound::Included(v) => *v,
            Bound::Excluded(v) => v.saturating_add(1),
            Bound::Unbounded => 0,
        };
        let end_exclusive = match range.end_bound() {
            Bound::Included(v) => v.saturating_add(1),
            Bound::Excluded(v) => *v,
            Bound::Unbounded => self.len(),
        };

        let mid = self.0.len();
        let slice_0 = start_inclusive.min(mid)..end_exclusive.min(mid);
        let slice_1 = (start_inclusive - slice_0.start)..(end_exclusive - slice_0.end);

        (slice_0, slice_1)
    }
}

impl<'a, T> Region<'a, T>
where
    T: Copy,
{
    /// Copies all elements in the region into `dest`, using a memcpy.
    ///
    /// The length of `dest` must be the same as `self`.
    ///
    /// If `T` does not implement `Copy`, use [`clone_to_slice`].
    ///
    /// # Panics
    /// This function will panic if `dest` has a different length to `self`.
    ///
    /// [`clone_to_slice`]: Region::clone_to_slice
    pub fn copy_to_slice(&self, dest: &mut [T]) {
        assert_eq!(
            self.len(),
            dest.len(),
            "destination and source slices have different lengths"
        );

        let (dest_0, dest_1) = dest.split_at_mut(self.0.len());
        dest_0.copy_from_slice(self.0);
        dest_1.copy_from_slice(self.1);
    }
}

impl<'a, T> Region<'a, T>
where
    T: Clone,
{
    /// Copies all elements in the region into `dest`.
    ///
    /// The length of `dest` must be the same as `self`.
    ///
    /// # Panics
    /// This function will panic if `dest` has a different length to `self`.
    pub fn clone_to_slice(&self, dest: &mut [T]) {
        assert_eq!(
            self.len(),
            dest.len(),
            "destination and source slices have different lengths"
        );

        let (dest_0, dest_1) = dest.split_at_mut(self.0.len());
        dest_0.clone_from_slice(self.0);
        dest_1.clone_from_slice(self.1);
    }
}

impl<'a, T> Default for Region<'a, T> {
    #[inline]
    fn default() -> Self {
        Region(Default::default(), Default::default())
    }
}

impl<'a, T> Index<usize> for Region<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.0.len() {
            &self.1[index - self.0.len()]
        } else {
            &self.0[index]
        }
    }
}

/// A region of two separate mutable slices that represent continuous data.
///
/// This is the mutable equivalent to [`Region`], allowing mutable access to the contained slices.
///
/// [`Region`]: crate::Region
#[derive(PartialEq, Eq, Hash)]
pub struct RegionMut<'a, T>(&'a mut [T], &'a mut [T]);

impl<'a, T> RegionMut<'a, T> {
    /// Creates a region with two mutable slices.
    ///
    /// The slices should represent continuous data, where `slice_0` goes before `slice_1`.
    #[inline]
    pub fn new(slice_0: &'a mut [T], slice_1: &'a mut [T]) -> Self {
        RegionMut(slice_0, slice_1)
    }

    /// Returns a non-mutable version of the region.
    #[inline]
    pub fn as_ref(&self) -> Region<T> {
        Region(self.0, self.1)
    }

    /// Returns `true` if the region has a length of 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }

    /// Returns the number of elements in the region.
    #[inline]
    pub fn len(&self) -> usize {
        self.as_ref().len()
    }

    /// Returns an iterator over the region.
    ///
    /// The iterator yields all items from start to end.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.as_ref().iter()
    }

    /// Returns an iterator that allows modifying each value.
    ///
    /// The iterator yields all items from start to end.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        [self.0.iter_mut(), self.1.iter_mut()].into_iter().flatten()
    }

    /// Returns the first contiguous slice in the region.
    ///
    /// The slice will not necessarily contain all data in the region.
    #[inline]
    pub fn contiguous(&self) -> &[T] {
        self.as_ref().contiguous()
    }

    /// Returns a mutable reference to the first contiguous slice in the region.
    ///
    /// The slice will not necessarily contain all data in the region.
    #[inline]
    pub fn contiguous_mut(&mut self) -> &mut [T] {
        if self.0.is_empty() {
            self.1
        } else {
            self.0
        }
    }

    /// Slices the region, returning a new region containing a part of the original.
    ///
    /// # Panics
    /// Panics if the `start` or `end` of the range is `>= region.len()`.
    #[inline]
    pub fn slice<R>(&self, range: R) -> Region<T>
    where
        R: RangeBounds<usize>,
    {
        self.as_ref().slice(range)
    }

    /// Slices the region, returning a new mutable region containing a part of the original.
    ///
    /// # Panics
    /// Panics if the `start` or `end` of the range is `>= region.len()`.
    pub fn slice_mut<R>(&mut self, range: R) -> RegionMut<T>
    where
        R: RangeBounds<usize>,
    {
        let (range_0, range_1) = self.as_ref().slice_ranges(range);

        let slice_0 = &mut self.0[range_0];
        let slice_1 = &mut self.1[range_1];

        RegionMut(slice_0, slice_1)
    }
}

impl<'a, T> RegionMut<'a, T>
where
    T: Copy,
{
    /// Copies all elements in the region into `dest`, using a memcpy.
    ///
    /// The length of `dest` must be the same as `self`.
    ///
    /// If `T` does not implement `Copy`, use [`clone_to_slice`].
    ///
    /// # Panics
    /// This function will panic if `dest` has a different length to `self`.
    ///
    /// [`clone_to_slice`]: RegionMut::clone_to_slice
    #[inline]
    pub fn copy_to_slice(&self, slice: &mut [T]) {
        self.as_ref().copy_to_slice(slice)
    }

    /// Copies all elements from `src` into the region, using a memcpy.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// If `T` does not implement `Copy`, use [`clone_from_slice`].
    ///
    /// # Panics
    /// This function will panic if `src` has a different length to `self`.
    ///
    /// [`clone_from_slice`]: RegionMut::clone_from_slice
    pub fn copy_from_slice(&mut self, src: &[T]) {
        assert_eq!(
            self.len(),
            src.len(),
            "destination and source slices have different lengths"
        );

        let (src_0, src_1) = src.split_at(self.0.len());
        self.0.copy_from_slice(src_0);
        self.1.copy_from_slice(src_1);
    }
}

impl<'a, T> RegionMut<'a, T>
where
    T: Clone,
{
    /// Copies all elements in the region into `dest`.
    ///
    /// The length of `dest` must be the same as `self`.
    ///
    /// # Panics
    /// This function will panic if `dest` has a different length to `self`.
    #[inline]
    pub fn clone_to_slice(&self, slice: &mut [T]) {
        self.as_ref().clone_to_slice(slice)
    }

    /// Copies all elements from `src` into the region.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// # Panics
    /// This function will panic if `src` has a different length to `self`.
    pub fn clone_from_slice(&mut self, src: &[T]) {
        assert_eq!(
            self.len(),
            src.len(),
            "destination and source slices have different lengths"
        );

        let (src_0, src_1) = src.split_at(self.0.len());
        self.0.clone_from_slice(src_0);
        self.1.clone_from_slice(src_1);
    }
}

impl<'a, T> Default for RegionMut<'a, T> {
    #[inline]
    fn default() -> Self {
        RegionMut(Default::default(), Default::default())
    }
}

impl<'a, T> Index<usize> for RegionMut<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.0.len() {
            &self.1[index - self.0.len()]
        } else {
            &self.0[index]
        }
    }
}

impl<'a, T> IndexMut<usize> for RegionMut<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= self.0.len() {
            &mut self.1[index - self.0.len()]
        } else {
            &mut self.0[index]
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Region, RegionMut};

    fn collect<T: Clone>(region: &Region<T>) -> Vec<T> {
        region.iter().cloned().collect()
    }

    #[test]
    fn region_default_is_empty() {
        assert!(Region::<u8>::default().is_empty());
        assert_eq!(Region::<u8>::default().len(), 0);
    }

    #[test]
    fn region_is_empty() {
        let empty_arr = [];
        let full_arr = [1, 2, 3];

        assert!(Region::new(&empty_arr, &empty_arr).is_empty());
        assert!(!Region::new(&full_arr, &full_arr).is_empty());
        assert!(!Region::new(&full_arr, &empty_arr).is_empty());
        assert!(!Region::new(&empty_arr, &full_arr).is_empty());
    }

    #[test]
    fn region_len() {
        let empty_arr = [];
        let full_arr = [1, 2, 3];

        assert_eq!(Region::new(&empty_arr, &empty_arr).len(), 0);
        assert_eq!(Region::new(&full_arr, &full_arr).len(), 6);
        assert_eq!(Region::new(&full_arr, &empty_arr).len(), 3);
        assert_eq!(Region::new(&empty_arr, &full_arr).len(), 3);
    }

    #[test]
    fn region_iter() {
        let empty_arr = [];
        let arr_1 = [1, 2, 3];
        let arr_2 = [4, 5, 6];

        assert_eq!(
            Region::new(&empty_arr, &empty_arr)
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![]
        );
        assert_eq!(
            Region::new(&arr_1, &arr_2)
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert_eq!(
            Region::new(&arr_1, &empty_arr)
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            Region::new(&empty_arr, &arr_2)
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![4, 5, 6]
        );
    }

    #[test]
    fn region_contiguous() {
        let empty_arr = [];
        let arr_1 = [1, 2, 3];
        let arr_2 = [4, 5, 6];

        assert!(Region::new(&empty_arr, &empty_arr).contiguous().is_empty());
        assert_eq!(Region::new(&arr_1, &arr_2).contiguous(), [1, 2, 3]);
        assert_eq!(Region::new(&arr_1, &empty_arr).contiguous(), [1, 2, 3]);
        assert_eq!(Region::new(&empty_arr, &arr_2).contiguous(), [4, 5, 6]);
    }

    #[test]
    fn region_slice() {
        let arr_1 = [1, 2, 3];
        let arr_2 = [4, 5, 6];
        let region = Region::new(&arr_1, &arr_2);

        assert_eq!(collect(&region.slice(..)), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(collect(&region.slice(1..)), vec![2, 3, 4, 5, 6]);
        assert_eq!(collect(&region.slice(3..)), vec![4, 5, 6]);
        assert_eq!(collect(&region.slice(5..)), vec![6]);
        assert_eq!(collect(&region.slice(6..)), vec![]);
        assert_eq!(collect(&region.slice(..5)), vec![1, 2, 3, 4, 5]);
        assert_eq!(collect(&region.slice(..3)), vec![1, 2, 3]);
        assert_eq!(collect(&region.slice(..1)), vec![1]);
        assert_eq!(collect(&region.slice(..0)), vec![]);
        assert_eq!(collect(&region.slice(..=5)), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(collect(&region.slice(1..3)), vec![2, 3]);
        assert_eq!(collect(&region.slice(2..4)), vec![3, 4]);
        assert_eq!(collect(&region.slice(3..5)), vec![4, 5]);
        assert_eq!(collect(&region.slice(0..6)), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn region_copy_to_slice() {
        let arr_1 = [1, 2, 3];
        let arr_2 = [4, 5, 6];
        let region = Region::new(&arr_1, &arr_2);

        let mut dest = [0, 0, 0, 0, 0, 0];
        region.copy_to_slice(&mut dest);

        assert_eq!(dest, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn region_clone_to_slice() {
        let arr_1 = [1, 2, 3];
        let arr_2 = [4, 5, 6];
        let region = Region::new(&arr_1, &arr_2);

        let mut dest = [0, 0, 0, 0, 0, 0];
        region.clone_to_slice(&mut dest);

        assert_eq!(dest, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn region_index() {
        let arr_1 = [1, 2, 3];
        let arr_2 = [4, 5, 6];
        let region = Region::new(&arr_1, &arr_2);

        assert_eq!(region[0], 1);
        assert_eq!(region[2], 3);
        assert_eq!(region[4], 5);
    }

    #[test]
    fn region_mut_as_ref() {
        let mut arr_1 = [1, 2, 3];
        let mut arr_2 = [4, 5, 6];
        let region = RegionMut::new(&mut arr_1, &mut arr_2);

        assert_eq!(collect(&region.as_ref()), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn region_mut_copy_from_slice() {
        let mut arr_1 = [1, 2, 3];
        let mut arr_2 = [4, 5, 6];
        let mut region = RegionMut::new(&mut arr_1, &mut arr_2);

        let src = [100, 101, 102, 103, 104, 105];
        region.copy_from_slice(&src);

        assert_eq!(
            collect(&region.as_ref()),
            vec![100, 101, 102, 103, 104, 105]
        );
        assert_eq!(region.contiguous(), [100, 101, 102]);
    }

    #[test]
    fn region_mut_clone_from_slice() {
        let mut arr_1 = [1, 2, 3];
        let mut arr_2 = [4, 5, 6];
        let mut region = RegionMut::new(&mut arr_1, &mut arr_2);

        let src = [100, 101, 102, 103, 104, 105];
        region.clone_from_slice(&src);

        assert_eq!(
            collect(&region.as_ref()),
            vec![100, 101, 102, 103, 104, 105]
        );
        assert_eq!(region.contiguous(), [100, 101, 102]);
    }

    #[test]
    fn region_mut_index() {
        let mut arr_1 = [1, 2, 3];
        let mut arr_2 = [4, 5, 6];
        let mut region = RegionMut::new(&mut arr_1, &mut arr_2);

        region[1] = 100;
        region[3] = 200;
        region[5] = 300;

        assert_eq!(collect(&region.as_ref()), vec![1, 100, 3, 200, 5, 300]);
    }
}
