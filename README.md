# mini-io-queue

[![Crates.io][crates-badge]][crates-url]
[![Docs.rs][docs-badge]][docs-url]
[![MIT licensed][mit-badge]][mit-url]
[![Build status][build-badge]][build-url]

[crates-badge]: https://img.shields.io/crates/v/mini-io-queue.svg
[crates-url]: https://crates.io/crates/mini-io-queue
[docs-badge]: https://img.shields.io/docsrs/mini-io-queue
[docs-url]: https://docs.rs/mini-io-queue/latest/mini-io-queue/
[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit-url]: https://github.com/cpdt/mini-io-queue/blob/master/LICENSE
[build-badge]: https://github.com/cpdt/mini-io-queue/workflows/Build/badge.svg
[build-url]: https://github.com/cpdt/mini-io-queue/actions?query=workflow%3ABuild+branch%3Amain

Fixed-length, allocation and lock-free, async I/O oriented single-producer single-consumer (SPSC) queues.

# Overview

This library provides several fixed-length queues for different use-cases, based around the same
core:
- `asyncio` is a generic async, thread-safe, lock-free queue using [futures]. Read
  operations can pause until data is available, and write operations can pause until space is
  returned. The queue can act as an efficient I/O buffer with [`futures::AsyncRead`] and
  [`futures::AsyncWrite`] implementations.
- `blocking` is a generic thread-safe queue. Read operations can block until data is
  available, and write operations can block until space is returned. The queue can act as an
  efficient I/O buffer with [`io::Read`] and [`io::Write`] implementations.
- `nonblocking` is a generic thread-safe, lock-free queue that is guaranteed to not block.
  It can act as an efficient I/O buffer when read and write speed is matched or no locks
  are available.

All queues have separate `Reader` and `Writer` ends which can be sent across threads. Queues are
designed with bulk operations in mind, so can safely be used with large read and write
operations, such as in a byte array I/O context.

The library also provides `Ring`, a low-level atomic ring buffer building block used to
implement the various queues available.

The library supports `no_std` with a reduced feature set, and is highly configurable. With the
default feature set, it does not require any dependencies.

# Examples

## Simple async example
```rust
use futures::executor::block_on;
use futures::join;
use mini_io_queue::asyncio::queue;

let (mut reader, mut writer) = queue(8);

let write_loop = async {
    for i in 0..16 {
        writer.write_all(&[i]).await.unwrap();
    }
};

let read_loop = async {
    for i in 0..16 {
        let mut buf = [0];
        reader.read_exact(&mut buf).await.unwrap();

        assert_eq!(buf[0], i);
    }
};

block_on(async { join!(write_loop, read_loop) });
```

## Blocking queue with a custom ring
```rust
use mini_io_queue::blocking::queue_from_parts;
use mini_io_queue::Ring;
use mini_io_queue::storage::{HeapBuffer, Storage};

// Create a queue with half of the underlying buffer in the read side.
let ring = Ring::new(10);
ring.advance_right(5);

let mut buffer = HeapBuffer::new(10);
buffer.slice_mut(0..5).copy_from_slice(&[1, 2, 3, 4, 5]);

let (mut reader, _) = queue_from_parts(ring, buffer);

for i in 1..=5 {
    let mut buf = [0];
    reader.read_exact(&mut buf).unwrap();
    assert_eq!(buf[0], i);
}
```

# License

Provided under the MIT license. Check [the LICENSE file](https://github.com/cpdt/mini-io-queue/blob/master/LICENSE) for details.

[futures]: https://docs.rs/futures/
[`futures::AsyncRead`]: https://docs.rs/futures/latest/futures/io/trait.AsyncRead.html
[`futures::AsyncWrite`]: https://docs.rs/futures/latest/futures/io/trait.AsyncWrite.html
[`io::Read`]: https://doc.rust-lang.org/std/io/trait.Read.html
[`io::Write`]: https://doc.rust-lang.org/std/io/trait.Write.html
[`alloc`]: https://doc.rust-lang.org/alloc/
[`std`]: https://doc.rust-lang.org/std/
