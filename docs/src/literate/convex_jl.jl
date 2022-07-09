#=
# Convex.jl Interface
Clarabel.jl implements support for [MathOptInterface](https://jump.dev/JuMP.jl/stable/moi/), and is therefore compatible with [Convex.jl](https://jump.dev/Convex.jl/stable/).   This allows you to describe and modify your optimisation problem with Convex.jl and use Clarabel as the backend solver.

## Setting Clarabel.jl Backend

You should construct your problem in the usual way in Convex.jl, and then solve using `Clarabel.Optimizer`, i.e. by calling `solve!` with
```julia
solve!(problem, Clarabel.Optimizer)
```
where `problem` is an object of type `Convex.Problem`.

## Convex.jl or JuMP?

Clarabel.jl supports both Convex.jl and JuMP via MathOptInterface.   Both packages are excellent and can make problem construction considerably easier than via the solver's native interface.

For problems with quadratic objective functions, JuMP is generally preferred when using Clarabel.jl since it will keep the quadratic function in the objective rather than reformulating the problem to a form with a linear cost and additional second-order cone constraints.   Clarabel.jl natively supports quadratic objectives and solve times are generally faster if this reformulation is avoided.

## Arbitrary Precision Arithmetic

Clarabel.jl supports arbitrary precision arithmetic for Convex.jl.   Here is the [Basic QP Example](@ref) implemented using `BigFloat` types.

=#

#hide setprecision(BigFloat,256)
using Clarabel, Convex

x = Variable(2)
objective = 3square(x[1]) + 2square(x[2]) - x[1] - 4x[2]
problem = minimize(objective; numeric_type = BigFloat)
problem.constraints = [x[1] == 2x[2]]
problem.constraints += [x >= -1; x <= 1]
solve!(problem, Clarabel.Optimizer{BigFloat}; silent_solver = false)
