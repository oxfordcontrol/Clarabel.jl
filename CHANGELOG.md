# Changelog

Changes for the Julia version of Clarabel are documented in this file.   For the Rust version, see [here](https://github.com/oxfordcontrol/clarabel.rs/CHANGELOG.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Version numbering in this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).  We aim to keep the core solver functionality and minor releases in sync between the Rust/Python and Julia implementations.   Small fixes that affect one implementation only may result in the patch release versions differing.

## [0.4.0] - 2023-25-02

### Changed 

- Internal fixes relating to initialization of iterates in symmetric cone problems.

- Numerical stability improvements for second order cone constraints. 

### Julia-specific changes

- Modification of the internal calls to QDLDL.jl to allow for direct assignment of parameters in AMD ordering.   This release requires QDLDL.jl v0.4.0.

- Makes Pardiso an optional dependency via Requires.jl.  To use Pardiso/MKL it is not necessary to import the Pardiso package directly before calling any part of the solver.  Fixes [#108](https://github.com/oxfordcontrol/Clarabel.jl/issues/108)


## [0.3.0] - 2022-09-13

### Changed 

- Implements support for exponential and power cones

- Numerical stability improvements

- Various bug fixes

## [0.2.0] - 2022-07-31

- Companion rust/python implementation released starting from this version.

- Ported all documentation to the common site [here](https://github.com/oxfordcontrol/ClarabelDocs)


## [0.1.0] - 2022-07-04

- Initial release



[0.4.0]: https://github.com/pyo3/pyo3/compare/v0.4.0...v0.3.0
[0.3.0]: https://github.com/pyo3/pyo3/compare/v0.3.0...v0.2.0
[0.2.0]: https://github.com/pyo3/pyo3/compare/v0.2.0...v0.1.0
[0.1.0]: https://github.com/PyO3/pyo3/tree/0.1.0
