# Getting Started

This guide describes the process of creating a Clarabel.jl model, populating its settings and problem data, solving the problem and obtaining and understanding results.  The description here relates to Clarabel's native API.  If you want to use `JuMP` or `Convex.jl` to model your  problem instead, see the [JuMP Interface](@ref) and [Convex.jl Interface](@ref) sections.

Clarabel.jl solves optimisation problems in the format:
```math
\begin{array}{ll} \text{minimize} & \textstyle{\frac{1}{2}}x^\top Px + q^\top x\\ \text{subject to} & Ax + s = b \\ & s \in \mathcal{K}, \end{array}
```

with decision variables ``x \in \mathbb{R}^n``, ``s \in \mathbb{R}^m`` and data matrices ``P=P^\top \succeq 0``, ``q \in \mathbb{R}^n``, ``A \in \mathbb{R}^{m \times n}``, and ``b \in \mathbb{R}^m``.  The convex cone ``\mathcal{K}``
is a composition of smaller convex cones ``\mathcal{K} = \mathcal{K}_1 \times \mathcal{K}_2  \dots \mathcal{K}_p``.   Equality conditions can be modelled in this format using the solver's ZeroCone type.   

## Making a Solver
The problem data, user settings and workspace variables are all stored in a top level `Solver` type. To get started define an empty Solver:
```julia
using Clarabel
model = Clarabel.Solver()
```
It is also possible to specify one or more solver configuration settings at creation time.   See the [Settings][@ref] section below.   

To initialize the solver with an optimisation problem we require three more things:
* The objective function, i.e. the matrix `P` and the vector `q` in ``\frac{1}{2}x^\top P x + q^\top x``
* The data matrix `A` and vector `b`, along with a description of the composite cone `\mathcal{K}` and the dimensions of its constituent pieces.
* A `Settings` object that specifies how Clarabel.jl solves the problem _(optional)_

## Settings

Solver settings are stored in a `Settings` object and can be modified by the user. To create a `Settings` object just call the constructor:

```julia
settings = Clarabel.Settings()
```

To adjust those values, you can pass options and parameters as a key-value pair to the constructor or edit the corresponding field afterwards. For example, if you want to disable verbose printing and set a 5 second time limit on the solver, you can use
```julia
settings = Clarabel.Settings(verbose = false, time_limit = 5)

# the following is equivalent
settings = Clarabel.Settings()
settings.verbose    = false
settings.time_limit = 5
```

## Objective Function
To set the objective function of your optimisation problem simply define the square positive semidefinite matrix ``P \in \mathrm{R}^{n\times n} `` and the vector ``q \in \mathrm{R}^{n}``. Clarabel.jl expects the `P` matrix to be supplied in sparse format.   The matrix `P` is assumed by the solver to be symmetric and only values in the upper triangular part of `P` are needed by the solver.

## Constraints
The Clarabel.jl interface expects constraints to be presented in the single vectorized form ``Ax + s = b, s \in \mathcal{K}``, where ``\mathcal{K} = \mathcal{K}_1 \times \dots \times \mathcal{K}_p`` and each ``\mathcal{K}_i`` is one of the  cones defined below:

Cone Type| Description
-----      |   :-----
`ZeroConeT`    | The set ``\{ 0 \}^{dim}`` that contains the origin
`NonnegativeConeT` | The nonnegative orthant ``\{ x \in \mathbb{R}^{dim} : x_i \ge 0, \forall i=1,\dots,\mathrm{dim} \}``
`SecondOrderConeT` | The second-order (Lorenz) cone ``\{ (t,x) \in \mathbb{R}^{dim}  :  \|x\|_2   \leq t \}``


Suppose that we have a problem with decision variable ``x \in \mathbb{R}^3`` and our constraints are:
* A single equality constraint ``x_1 + x_2 - x_3 = 1``.   
* A pair of inequalities such that ``x_2`` and ``x_3`` are each less than 2.
* A second order cone constraint on the 3-dimensional vector ``x``.   

We can then define our constraint data as

```julia
using SparseArrays

# equality constraint
Aeq = [1 1 -1]
beq = [1]

# inequality constraint
Aineq = [0 1 0;
         0 0 1]
bineq = [2,2]

# SOC constraint
Asoc = -I(3)
bsoc = [0,0,0]

#Clarabel.jl constraint data
A = sparse([Aeq; Aineq; Asoc])
b = [beq;bineq;bsoc]
```

Clarabel.jl expects to receive a vector of cone specifications.  For the above constraints we  should also define
```julia
#Clarabel.jl cone specification
cones = [Clarabel.ZeroConeT(1), Clarabel.NonnegativeConeT(2), SecondOrderConeT(3)]
```

!!! note
    The cones `cones' should be of type `Vector{Clarabel.SupportedCone}`, and your input vector `b` should be compatible with the sum of the cone dimensions.



!!! note
    Note carefully the signs in the above example.   The inequality condition is ``A_{ineq} x \le b_{ineq}``, which is equivalent to ``A_{ineq} x + s = b_{ineq}`` with ``s \ge 0``, i.e. ``s`` in the Nonnegative cone.    The SOC condition is ``x \in \mathcal{K}_{SOC}``, or equivalently ``-x + s = 0`` with ``s \in \mathcal{K}_{SOC}``.


## Adding problem data
Once the objective function and an array of constraints have been defined, you can provide the solver with problem data using
```julia
Clarabel.setup!(solver, P, q, A, b, cones, settings)
```
This takes an internal copy of all data parameters and initializes internal variables and other objects in the solver.  The final `settings` argument is optional.


## Solving
Now you can solve it using:
```julia
result = Clarabel.solve!(solver)
```

## Results

Once the solver algorithm terminates you can inspect the solution using the `solution` object.   The primal solution will be in `solution.x` and the dual solution in `solution.z`. The outcome of the solve is specified in `solution.status` and will be one of the following :

### Status Codes


Status Code  | Description
---  | :---
UNSOLVED            |  Default value, only occurs prior to calling `Clarabel.solve!`
SOLVED              |  Solution found
PRIMAL_INFEASIBLE   |  Problem is primal infeasible
DUAL_INFEASIBLE     |  Problem is dual infeasible
MAX_ITERATIONS      |  Solver halted after reaching iteration limit
MAX_TIME            |  Solver halted after reaching time limit

The total solution time (include combined `setup!` and `solve!` times) is given in `solution.solve_time`.   Detailed information about the solve time and memory allocation can be found in the solver's `timer` field.

!!! warning
    Be careful to retrieve solver solutions from the `solution` that is returned by the solver, or directly from a `solver` object from the `solver.solution` field.   Do *not* use the `solver.variables`, since these have both homogenization and equilibration scaling applied and therefore do *not* solve the optimization problem posed to the solver.

### Settings

The full set of user configurable solver settings are listed in the [API Reference](@ref api-settings)
