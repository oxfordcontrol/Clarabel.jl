__Clarabel.jl__ is a Julia implementation of an Interior Point numerical solver for convex optimization problems using a homogeneous embedding.  Clarabel.jl solves the following problem:

```math
\begin{array}{ll} \text{minimize} & \textstyle{\frac{1}{2}}x^\top Px + q^\top x\\ \text{subject to} & Ax + s = b \\ & s \in \mathcal{K},
\end{array}
```

with decision variables ``x \in \mathbb{R}^n``, ``s \in \mathbb{R}^m`` and data matrices ``P=P^\top \succeq 0``, ``q \in \mathbb{R}^n``, ``A \in \mathbb{R}^{m \times n}``, and ``b \in \mathbb{R}^m``. The convex set ``\mathcal{K}`` is a composition of convex cones.

## Features

* __Versatile__: COSMO solves linear programs, quadratic programs, second-order cone programs.  Future versions will provide support for semidefinite programs and problems involving exponential and power cones
* __Quadratic objectives__: Unlike most interior point solvers based on the standard homogeneous self-dual embedding (HSDE), Clarabel.jl handles quadratic objective without requiring reformulation of the objective.   It can therefore be significantly faster than other HSDE-based solvers for problems with quadratic objective functions.
* __Infeasibility detection__: Infeasible problems are detected using using a homogeneous embedding method.
* __JuMP / Convex.jl support__: We provide an interface to MathOptInterface (MOI), which allows you to describe your problem in [JuMP](https://github.com/JuliaOpt/JuMP.jl) and [Convex.jl](https://github.com/JuliaOpt/Convex.jl).
* __Arbitrary precision types__: You can solve problems with any floating point precision (In a future release).
* __Open Source__: Our code is available on [GitHub](https://github.com/oxfordcontrol/Clarabel.jl) and distributed under the Apache 2.0 Licence

## Installation
COSMO can be installed using the Julia package manager for Julia `v1.0` and higher. Inside the Julia REPL, type `]` to enter the Pkg REPL mode then run

`pkg> add Clarabel`

If you want to install the latest version from the github repository run

`pkg> add Clarabel#main`

## Credits

The following people are involved in the development of Clarabel.jl:
* [Paul Goulart](http://users.ox.ac.uk/~engs1373/) (main development, maths and algorithms)
* [Yuwen Chen](https://markcannon.github.io) (maths and algorithms)
\*all contributors are affiliated with the [University of Oxford](http://www2.eng.ox.ac.uk/control).

If this project is useful for your work please consider
* [Citing](citing.md) the relevant papers
* Leaving a star on the [GitHub repository](https://github.com/oxfordcontrol/Clarabel.jl)


## Licence
Clarabel.jl is licensed under the Apache License 2.0. For more details click [here](https://github.com/oxfordcontrol/Clarabel.jl/blob/master/LICENSE.md).
