# `mcintir` - Monte Carlo integrators in Rust

[![Rust](https://github.com/cschwan/mcintir/workflows/Rust/badge.svg)](https://github.com/cschwan/mcintir/actions?query=workflow%3ARust)
[![Gitter](https://badges.gitter.im/mcintir/support.svg)](https://gitter.im/mcintir/support?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


`mcintir`—pronounced like the name MacIntyre in English—is a library, which
offers several Monte Carlo integrators. The following are implemented:

  - PLAIN or naive Monte Carlo integration,

This library is a rewrite of the C++ library [`hep-mc`][hep-mc] and will
supersede it.

******
## Building

Building `mcintir` requires the [`Rust`](https://www.rust-lang.org/) toolchain to be available. Instructions on how to set it up can be found [here](https://www.rust-lang.org/tools/install).

Once this is done, `mcintir` can be compiled by simply running

```sh
> cargo b [--release]
```

******
## Usage

To demonstrate the usage of the library, we provide examples in the [`examples`](examples) directory.

******
## Documentation

The generation of the API documentation differs slightly from the usual `cargo doc` due to the presence of `LaTeX` formulas. To generate and view the documentation in the browser, simply
type

```sh
> 
RUSTDOCFLAGS="--html-in-header katex-header.html" cargo doc --no-deps --open
```

******
## Tests

`Integration` and `unit` tests can be run by typing

```sh
> cargo t [--release]
```

******
## Benchmarks

To run the benchmarks, first install `cargo-criterion` as follows

```sh
> cargo install cargo-criterion
```

After that, running the benchmarks is triggered by the command

```sh
> cargo criterion
```

The results of the benchmark can be viewed in the browser by typing

```sh
> firefox ./target/criterion/reports/index.html
```

******
## Discussion

You can find various channels for discussion at

- <https://gitter.im/mcintir/support> for user support,
- <https://gitter.im/mcintir/development> for development, and
- <https://gitter.im/mcintir/community> for everything else not fitting the
  above two categories.

[hep-mc]: https://github.com/cschwan/hep-mc
