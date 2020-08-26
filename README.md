[![Rust](https://github.com/cschwan/mcintir/workflows/Rust/badge.svg)](https://github.com/cschwan/mcintir/actions?query=workflow%3ARust)

# `mcintir` - Monte Carlo integrators in Rust

`mcintir`—pronounced like the name MacIntyre in English—is a library, which
offers several Monte Carlo integrators. The following are implemented:

  - PLAIN or naive Monte Carlo integration,

This library is a rewrite of the C++ library [`hep-mc`][hep-mc] and will
supersede it.

# Building

`mcintir` requires the Rust compiler to be available. 
Instructions on how to install it can be found at on the 
`[Rust website](https://www.rust-lang.org/tools/install)`.

The library can be build by running

```sh
> cargo build [--release]
```

The `release` option can be used to optimize the performance at the cost of 
larger compile times.

# Examples

`mcintir` comes with a set of example programs showcasing the usage of the library.

They can be run with

```sh
> cargo build [--release] --examples plain
```

# Tests

Unit, integration and doc tests can be run with 

```sh
cargo test [--release]
```

# Documentation

`mcinitir`'s documentation features `LaTeX` formulas and therefore its generation is slightly different from the 
usual case. It can be generated through

```sh
> RUSTDOCFLAGS="--html-in-header katex-header.html" cargo doc [--release] [--no-deps] [--open]
```

# Discussion

You can find various channels for discussion at

- <https://gitter.im/mcintir/support> for user support,
- <https://gitter.im/mcintir/development> for development, and
- <https://gitter.im/mcintir/community> for everything else not fitting the
  above two categories.

[hep-mc]: https://github.com/cschwan/hep-mc
