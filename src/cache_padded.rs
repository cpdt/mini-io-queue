//! Vendored from <https://github.com/smol-rs/cache-padded> (commit 8b45665).
//!
//! Prevent false sharing by padding and aligning to the length of a cache line.
//!
//! In concurrent programming, sometimes it is desirable to make sure commonly accessed shared data
//! is not all placed into the same cache line. Updating an atomic value invalides the whole cache
//! line it belongs to, which makes the next access to the same cache line slower for other CPU
//! cores. Use CachePadded to ensure updating one piece of data doesn't invalidate other cached
//! data.

use core::fmt;
use core::ops::{Deref, DerefMut};

/// Pads and aligns data to the length of a cache line.
// Starting from Intel's Sandy Bridge, spatial prefetcher is now pulling pairs of 64-byte cache
// lines at a time, so we have to align to 128 bytes rather than 64.
//
// Sources:
// - https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf
// - https://github.com/facebook/folly/blob/1b5288e6eea6df074758f877c849b6e73bbb9fbb/folly/lang/Align.h#L107
//
// ARM's big.LITTLE architecture has asymmetric cores and "big" cores have 128-byte cache line size.
//
// Sources:
// - https://www.mono-project.com/news/2016/09/12/arm64-icache/
//
// powerpc64 has 128-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_ppc64x.go#L9
#[cfg_attr(
any(
target_arch = "x86_64",
target_arch = "aarch64",
target_arch = "powerpc64",
),
repr(align(128))
)]
// arm, mips, mips64, and riscv64 have 32-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_arm.go#L7
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_mips.go#L7
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_mipsle.go#L7
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_mips64x.go#L9
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_riscv64.go#L7
#[cfg_attr(
any(
target_arch = "arm",
target_arch = "mips",
target_arch = "mips64",
target_arch = "riscv64",
),
repr(align(32))
)]
// s390x has 256-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_s390x.go#L7
#[cfg_attr(target_arch = "s390x", repr(align(256)))]
// x86 and wasm have 64-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/dda2991c2ea0c5914714469c4defc2562a907230/src/internal/cpu/cpu_x86.go#L9
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_wasm.go#L7
//
// All others are assumed to have 64-byte cache line size.
#[cfg_attr(
not(any(
target_arch = "x86_64",
target_arch = "aarch64",
target_arch = "powerpc64",
target_arch = "arm",
target_arch = "mips",
target_arch = "mips64",
target_arch = "riscv64",
target_arch = "s390x",
)),
repr(align(64))
)]
#[derive(Clone, Copy, Default, Hash, PartialEq, Eq)]
pub struct CachePadded<T>(T);

impl<T> CachePadded<T> {
    /// Pads and aligns a piece of data to the length of a cache line.
    ///
    /// # Examples
    ///
    /// ```
    /// use cache_padded::CachePadded;
    ///
    /// let padded = CachePadded::new(1);
    /// ```
    pub const fn new(t: T) -> CachePadded<T> {
        CachePadded(t)
    }
}

impl<T> Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for CachePadded<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: fmt::Debug> fmt::Debug for CachePadded<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CachePadded").field(&self.0).finish()
    }
}

impl<T> From<T> for CachePadded<T> {
    fn from(t: T) -> Self {
        CachePadded::new(t)
    }
}
