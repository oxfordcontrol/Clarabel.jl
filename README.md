
<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/oxfordcontrol/ClarabelDocs/blob/main/docs/src/assets/logo-banner-dark-jl.png" width=60%>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/oxfordcontrol/ClarabelDocs/blob/main/docs/src/assets/logo-banner-light-jl.png" width=60%>
  <img alt="Clarabel.jl logo" src="https://github.com/oxfordcontrol/ClarabelDocs/blob/main/docs/src/assets/logo-banner-light-jl.png" height="25">
</picture>
<h1 align="center" margin=0px>
Interior Point Conic Optimization for Julia
</h1>
   <a href="https://github.com/oxfordcontrol/Clarabel.jl/actions"><img src="https://github.com/oxfordcontrol/Clarabel.jl/workflows/ci/badge.svg?branch=main"></a>
  <a href="https://codecov.io/gh/oxfordcontrol/Clarabel.jl"><img src="https://codecov.io/gh/oxfordcontrol/Clarabel.jl/branch/main/graph/badge.svg"></a>
  <a href="https://oxfordcontrol.github.io/ClarabelDocs/stable"><img src="https://img.shields.io/badge/Documentation-stable-purple.svg"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://github.com/oxfordcontrol/Clarabel.jl/releases"><img src="https://img.shields.io/badge/Release-v0.9.0-blue.svg"></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#license-">License</a> •
  <a href="https://oxfordcontrol.github.io/ClarabelDocs/stable">Documentation</a>
</p>

__Clarabel.jl__ is a Julia implementation of an interior point numerical solver for convex optimization problems using a novel homogeneous embedding.  Clarabel.jl solves the following problem:

$$
\begin{array}{r}
\text{minimize} & \frac{1}{2}x^T P x + q^T x\\\\[2ex]
\text{subject to} & Ax + s = b \\\\[1ex]
        & s \in \mathcal{K}
\end{array}
$$

with decision variables
$x \in \mathbb{R}^n$,
$s \in \mathbb{R}^m$
and data matrices
$P=P^\top \succeq 0$,
$q \in \mathbb{R}^n$,
$A \in \mathbb{R}^{m \times n}$, and
$b \in \mathbb{R}^m$.
The convex set $\mathcal{K}$ is a composition of convex cones.


__For more information see the Clarabel Documentation ([stable](https://oxfordcontrol.github.io/ClarabelDocs/stable) |  [dev](https://oxfordcontrol.github.io/ClarabelDocs/dev)).__

Clarabel is also available in a Rust implementation with additional language interfaces.  See [here](https://github.com/oxfordcontrol/Clarabel.rs).

## Features

* __Versatile__: Clarabel.jl solves linear programs (LPs), quadratic programs (QPs), second-order cone programs (SOCPs) and semidefinite programs (SDPs). It also solves problems with exponential, power cone and generalized power cone constraints.
* __Quadratic objectives__: Unlike interior point solvers based on the standard homogeneous self-dual embedding (HSDE), Clarabel.jl handles quadratic objectives without requiring any epigraphical reformulation of the objective.   It can therefore be significantly faster than other HSDE-based solvers for problems with quadratic objective functions.
* __Infeasibility detection__: Infeasible problems are detected using a homogeneous embedding technique.
* __JuMP / Convex.jl support__: We provide an interface to [MathOptInterface](https://jump.dev/JuMP.jl/stable/moi/) (MOI), which allows you to describe your problem in [JuMP](https://github.com/jump-dev/JuMP.jl) and [Convex.jl](https://github.com/jump-dev/Convex.jl).
* __Arbitrary precision types__: You can solve problems with any floating point precision, for example, Float32 or Julia's BigFloat type, using either the native interface, or via MathOptInterface / Convex.jl.
* __Open Source__: Our code is available on [GitHub](https://github.com/oxfordcontrol/Clarabel.jl) and distributed under the Apache 2.0 License

## Installation
- __Clarabel.jl__ can be added via the Julia package manager (type `]`): `pkg> add Clarabel`

## Citing
```
@misc{Clarabel_2024,
      title={Clarabel: An interior-point solver for conic programs with quadratic objectives}, 
      author={Paul J. Goulart and Yuwen Chen},
      year={2024},
      eprint={2405.12762},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```

## License 🔍
This project is licensed under the Apache License  2.0 - see the [LICENSE.md](https://github.com/oxfordcontrol/Clarabel.jl/blob/main/LICENSE.md) file for details.

