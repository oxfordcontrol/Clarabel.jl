# Changelog

Changes for the Julia version of Clarabel are documented in this file. For the Rust version, see [here](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/CHANGELOG.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Version numbering in this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).  We aim to keep the core solver functionality and minor releases in sync between the Rust/Python and Julia implementations. Small fixes that affect one implementation only may result in the patch release versions differing.

## [0.9.0] - 2024-06-06

### Changed
- Read/write problems to JSON files [#169](https://github.com/oxfordcontrol/Clarabel.jl/pull/169)

### Julia-specific changes
- Fix crash on systems with 32-bit blas [#171]([https://github.com/oxfordcontrol/Clarabel.jl/pull/16](https://github.com/oxfordcontrol/Clarabel.jl/issues/171)


## [0.8.1] - 2024-22-05
### Changed 

- minor documentation updates

## [0.8.0] - 2024-21-05
### Changed 

- implements chordal decomposition for PSD cones [#164](https://github.com/oxfordcontrol/Clarabel.jl/pull/164)
- updates scaling bounds [#165](https://github.com/oxfordcontrol/Clarabel.jl/pull/165)

### Julia specific changes
* Adds support for several 3rd party linear solvers -- HSL MA57, Panua Pardiso and MKL Pardiso [#166](https://github.com/oxfordcontrol/Clarabel.jl/pull/166)

## [0.7.1] - 2024-29-02
### Changed 

- Fixes a panic / crash condition in PSD scaling step [#78](https://github.com/oxfordcontrol/Clarabel.rs/pull/78)

### Julia specific changes
- Fix for LDL solvers using lower triangular format [#155](https://github.com/oxfordcontrol/Clarabel.jl/issues/155)

## [0.7.0] - 2024-26-02
### Changed 

- Solution output reports dual objective values.  Infeasible problems report NaN. 
- Solver now supports problems with a mix of PSD and nonsymmetric cones. 
- Added methods for updating problem data without reallocating memory.  
- Bug fix enforcing scaling limits in equilibration.  
- Bug fix in infeasibility detection. 

See [Rust version changelog](https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.6.0...v0.7.0) for related issues.

## [0.6.0] - 2023-20-09
### Changed 

- Introduces support for the generalized power cone. 
- Implements stability and speed improvements for SOC problems.  SOCs with dimension less than or equal to 4 are now treated as special cases with dense Hessian blocks.
- Fixes bad initialization point for non-quadratic objectives 
- Improved convergence speed for QPs with no constraints or only ZeroCone constraints.
- Internal code restructuring for cones with sparsifiable Hessian blocks.

### Julia specific changes
- Fixed a type error for Float32 problems [#135](https://github.com/oxfordcontrol/Clarabel.jl/issues/135)
- Update to ScaledPSDCone handling in MOI interface [#141](https://github.com/oxfordcontrol/Clarabel.jl/issues/141)

## [0.5.1] - 2023-02-06
### Changed 
Fixes convergence edge case in KKT direct solve iterative refinement.
### Julia specific changes
Updates to MOI interface to support scaled PSD cones directly [#131](https://github.com/oxfordcontrol/Clarabel.jl/issues/131) and to add missing termination status codes [#132](https://github.com/oxfordcontrol/Clarabel.jl/issues/132)

## [0.5.0] - 2023-25-04
### Changed 

This version ports support for PSD cones from the Julia version to Rust, with internal supporting modifications to both versions to keep implementations synchronized.
### Julia specific changes

- Julia package now uses SnoopPrecompile to reduce load times.  Load times will be faster in particular when using Julia versions 1.9 onwards, but code remains backwards compatible to older versions.  Removed some dependencies in favor of lighter weight ones in support.  Fixes [#120](https://github.com/oxfordcontrol/Clarabel.jl/issues/120)

- Solver now allows SDP cone constraints with dimension 0.

- Adds BigFloat support for SDPs.


## [0.4.1] - 2023-08-03
### Changed 

Added optional feature to remove inequality constraints with very large upper bounds.   This feature is enabled by default but can be turned off using the `presolve_enable` setting.  

Bug fix in equilibration for NN and zero cones.
### Julia specific changes

Internal implementation of composite cone logic updated to more closely match the rust version.

Internal modifications to SDP cone implementation to reduce allocations.
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

[0.9.0]: https://github.com/oxfordcontrol/Clarabel.jl/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/oxfordcontrol/Clarabel.jl/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/oxfordcontrol/Clarabel.jl/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/oxfordcontrol/Clarabel.jl/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/oxfordcontrol/Clarabel.jl/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/oxfordcontrol/Clarabel.jl/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/oxfordcontrol/Clarabel.jl/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/oxfordcontrol/Clarabel.jl/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/oxfordcontrol/Clarabel.jl/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/oxfordcontrol/Clarabel.jl/compare/v0.4.0...v0.3.0
[0.3.0]: https://github.com/oxfordcontrol/Clarabel.jl/compare/v0.3.0...v0.2.0
[0.2.0]: https://github.com/oxfordcontrol/Clarabel.jl/compare/v0.2.0...v0.1.0
[0.1.0]: https://github.com/oxfordcontrol/Clarabel.jl/tree/0.1.0
