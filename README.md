
<h1 align="center" margin=0px>
  <img src="https://github.com/oxfordcontrol/Clarabel.jl/blob/main/docs/src/assets/logo.png" width=70%>
</h1>
<p align="center">
   <a href="https://github.com/oxfordcontrol/Clarabel.jl/actions"><img src="https://github.com/oxfordcontrol/Clarabel.jl/workflows/ci/badge.svg?branch=master"></a>
  <a href="https://codecov.io/gh/oxfordcontrol/Clarabel.jl"><img src="https://codecov.io/gh/oxfordcontrol/Clarabel.jl/branch/master/graph/badge.svg"></a>
  <a href="https://oxfordcontrol.github.io/Clarabel.jl/stable"><img src="https://img.shields.io/badge/Documentation-stable-purple.svg"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://github.com/oxfordcontrol/Clarabel.jl/releases"><img src="https://img.shields.io/badge/Release-v0.1.0-blue.svg"></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="NEWS.md">News</a> •
  <a href="#license-">License</a>
</p>

__Clarabel.jl__ is a Julia implementation of an interior point numerical solver for convex optimization problems using a novel homogeneous embedding.  Clarabel.jl solves the following problem:

```math
\begin{array}{ll} \text{minimize} & \textstyle{\frac{1}{2}}x^\top Px + q^\top x\\ \text{subject to} & Ax + s = b \\ & s \in \mathcal{K},
\end{array}
```

with decision variables ``x \in \mathbb{R}^n``, ``s \in \mathbb{R}^m`` and data matrices ``P=P^\top \succeq 0``, ``q \in \mathbb{R}^n``, ``A \in \mathbb{R}^{m \times n}``, and ``b \in \mathbb{R}^m``. The convex set ``\mathcal{K}`` is a composition of convex cones.


__For more information see the Clarabel.jl Documentation ([stable](https://oxfordcontrol.github.io/Clarabel.jl/stable) |  [dev](https://oxfordcontrol.github.io/Clarabel.jl/dev)).__

## Features

* __Versatile__: Clarabel.jl solves linear programs (LPs), quadratic programs (QPs) and second-order cone programs (SOCPs).  Future versions will provide support for semidefinite programs and problems involving exponential and power cones.
* __Quadratic objectives__: Unlike interior point solvers based on the standard homogeneous self-dual embedding (HSDE), Clarabel.jl handles quadratic objective without requiring any epigraphical reformulation of the objective.   It can therefore be significantly faster than other HSDE-based solvers for problems with quadratic objective functions.
* __Infeasibility detection__: Infeasible problems are detected using using a homogeneous embedding technique.
* __JuMP / Convex.jl support__: We provide an interface to [MathOptInterface](https://jump.dev/JuMP.jl/stable/moi/) (MOI), which allows you to describe your problem in [JuMP](https://github.com/JuliaOpt/JuMP.jl) and [Convex.jl](https://github.com/JuliaOpt/Convex.jl).
* __Arbitrary precision types__: You can solve problems with any floating point precision, e.g. Float32 or Julia's BigFloat type, using either the native interface, or via MathOptInterface / Convex.jl.
* __Open Source__: Our code is available on [GitHub](https://github.com/oxfordcontrol/Clarabel.jl) and distributed under the Apache 2.0 Licence

## Installation
- `Clarabel.jl` can be added via the Julia package manager (type `]`): `pkg> add Clarabel`


## Licence 🔍
This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.md) file for details.
