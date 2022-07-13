#=
# Arbitrary Precision Arithmetic

Clarabel.jl supports the use of arbitrary precision floating-point types, including Julia's `BigFloat` type.  To use this feature you must specify all of your problem data using a common floating point type and explictly create Clarabel.Solver and (optional) Clarabel.Settings objects of the the same type.

Start by creating the solver and settings with the desired precision:
=#

using Clarabel, LinearAlgebra, SparseArrays

settings = Clarabel.Settings{BigFloat}(
            verbose = true,
            direct_kkt_solver = true,
            direct_solve_method = :qdldl)

solver   = Clarabel.Solver{BigFloat}()

#=
### Objective and constraint data

We next put the objective function into the standard Clarabel.jl form.   Here we use the same problem data as in the [Basic QP Example](@ref), but in `BigFloat` format :
=#

P = sparse(BigFloat[3. 0.;0. 2.].*2)
q = BigFloat[-1., -4.]
A = sparse(
    BigFloat[1. -2.;    #<-- LHS of equality constraint
             1.  0.;    #<-- LHS of inequality constraint (upper bound)
             0.  1.;    #<-- LHS of inequality constraint (upper bound)
            -1.  0.;    #<-- LHS of inequality constraint (lower bound)
             0. -1.;    #<-- LHS of inequality constraint (lower bound)
    ])
b = [zero(BigFloat);    #<-- RHS of equality constraint
     ones(BigFloat,4)   #<-- RHS of inequality constraints
    ]

cones =
    [Clarabel.ZeroConeT(1),           #<--- for the equality constraint
     Clarabel.NonnegativeConeT(4)]    #<--- for the inequality constraints

nothing  #hide

# You can optionally set the global precision of Julia's BigFloat type before solving
setprecision(BigFloat,128)
nothing  #hide

# Finally we can set up the problem in the usual way and solve

Clarabel.setup!(solver, P, q, A, b, cones, settings)
result = Clarabel.solve!(solver)

#then retrieve the solution

result.x

# Notice that the above would fail if the default solver was used, because Clarabel.jl uses Float64 by default
Clarabel.Solver()

#=
!!! warning
    For arbitrary precision arithmetic using `BigFloat` types you must select an internal linear solver
    within Clarabel.jl that supports it.   We recommend that you use the [QDLDL.jl](https://github.com/osqp/QDLDL.jl) package for such problems,
    and configure it as the linear solver by setting both `direct_kkt_solver = true` and `direct_solve_method = :qdldl` in the Settings object.

## With Convex.jl / JuMP

Clarabel.jl also supports arbitrary precision arithmetic through Convex.jl.   See the example in the [Convex.jl Interface](@ref) section.


!!! note
    `JuMP` does not currently support arbitrary precision. However, if you want to use `Clarabel` directly with `MathOptInterface`, you can use: `Clarabel.Optimizer{<: AbstractFloat}` as your optimizer.  As above, the problem data precision of your MathOptInterface-model must agree with the optimizer's precision.
=#
