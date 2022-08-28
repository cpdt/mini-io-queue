//! Fixed-length, allocation and lock-free, async I/O oriented single-producer single-consumer
//! (SPSC) queues.
//!
//! # Overview
//! This library provides several fixed-length queues for different use-cases, based around the same
//! core:
//!  - [`asyncio`] is a generic async, thread-safe, lock-free queue using [futures]. Read
//!    operations can pause until data is available, and write operations can pause until space is
//!    returned. The queue can act as an efficient I/O buffer with [`futures::AsyncRead`] and
//!    [`futures::AsyncWrite`] implementations.
//!  - [`blocking`] is a generic thread-safe queue. Read operations can block until data is
//!    available, and write operations can block until space is returned. The queue can act as an
//!    efficient I/O buffer with [`io::Read`] and [`io::Write`] implementations.
//!  - [`nonblocking`] is a generic thread-safe, lock-free queue that is guaranteed to not block.
//!    It can act as an efficient I/O buffer when read and write speed is matched or no locks
//!    are available.
//!
//! All queues have separate `Reader` and `Writer` ends which can be sent across threads. Queues are
//! designed with bulk operations in mind, so can safely be used with large read and write
//! operations, such as in a byte array I/O context.
//!
//! The library also provides [`Ring`], a low-level atomic ring buffer building block used to
//! implement the various queues available.
//!
//! The library supports `no_std` with a reduced feature set, and is highly configurable. With the
//! default feature set, it does not require any dependencies.
//!
//! # Features
//!
//!  - `asyncio` - enables the [`asyncio`] queue, using the [`futures`] library. Requires [`alloc`].
//!  - `blocking` - enables the [`blocking`] queue. Requires [`std`] (as this queue uses locks).
//!  - `nonblocking` - enables the [`nonblocking`] queue. Requires [`alloc`].
//!  - `heap-buffer` - enables [`HeapBuffer`], which allocates queue storage on the heap. Requires [`alloc`].
//!  - `stack-buffer` - enables [`StackBuffer`], which allocates queue storage on the stack.
//!  - `std-io` - enables implementations for standard I/O traits ([`io::Read`], [`io::Write`], and
//!    [`futures::AsyncRead`] and [`futures::AsyncWrite`]). Requires [`std`].
//!
//! # Examples
//! ## Simple async example
//! ```
//! use futures::executor::block_on;
//! use futures::join;
//! use mini_io_queue::asyncio::queue;
//!
//! let (mut reader, mut writer) = queue(8);
//!
//! let write_loop = async {
//!     for i in 0..16 {
//!         writer.write_all(&[i]).await.unwrap();
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
//! ## Blocking queue with a custom ring
//! ```
//! use mini_io_queue::blocking::queue_from_parts;
//! use mini_io_queue::Ring;
//! use mini_io_queue::storage::{HeapBuffer, Storage};
//!
//! // Create a queue with half of the underlying buffer in the read side.
//! let ring = Ring::new(10);
//! ring.advance_right(5);
//!
//! let mut buffer = HeapBuffer::new(10);
//! buffer.slice_mut(0..5).copy_from_slice(&[1, 2, 3, 4, 5]);
//!
//! let (mut reader, _) = queue_from_parts(ring, buffer);
//!
//! for i in 1..=5 {
//!     let mut buf = [0];
//!     reader.read_exact(&mut buf).unwrap();
//!
//!     assert_eq!(buf[0], i);
//! }
//! ```
//!
//! [`io::Read`]: std::io::Read
//! [`io::Write`]: std::io::Write
//! [`HeapBuffer`]: storage::HeapBuffer
//! [`StackBuffer`]: storage::StackBuffer

#![cfg_attr(not(any(feature = "std")), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs, missing_debug_implementations)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::private_intra_doc_links)]

extern crate core;

#[cfg(feature = "alloc")]
extern crate alloc;

mod cache_padded;
mod region;
mod ring;
pub mod storage;

#[cfg(feature = "asyncio")]
#[cfg_attr(docsrs, doc(cfg(feature = "asyncio")))]
pub mod asyncio;

#[cfg(feature = "nonblocking")]
#[cfg_attr(docsrs, doc(cfg(feature = "nonblocking")))]
pub mod nonblocking;

#[cfg(feature = "blocking")]
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
pub mod blocking;

pub use self::region::*;
pub use self::ring::*;
