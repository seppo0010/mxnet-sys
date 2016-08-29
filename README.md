# mxnet Rust Bindings

The `mxnet-sys` crate provides Rust low-level bindings to the `mxnet` C API.
For a higher-level API, please see the [`mxnet-rs`][mxnet-rs] crate.

## Dependencies

To use `mxnet-sys`, you'll need to build and install `libmxnet.so` where
Cargo can find it. For example, copy `libmxnet.so` to `/usr/local/lib`.

For details on how to build `libmxnet.so`, please see the [mxnet][] project.

## License

Distributed under the [ISC License][license].

[mxnet]: https://github.com/dmlc/mxnet
[mxnet-rs]: https://github.com/jakeleeme/mxnet-rs
[license]: LICENSE.txt
