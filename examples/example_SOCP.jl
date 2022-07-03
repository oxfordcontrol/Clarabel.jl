#=
# Basic SOCP Example

Suppose that we want to solve the following 2-dimensional optimization problem:

$$
\begin{array}{ll} \text{minimize} & x_2^2\\[2ex]
\text{subject to} &  \left\|\begin{pmatrix} 2x_1 \\ x_2 \end{pmatrix}
- \begin{pmatrix} 2 \\ 2 \end{pmatrix}\right\|_2 \le 1
\end{array}
$$

In this example we will see how to solve this problem both natively in Clarabel.jl
and also by solving with Clarabel.jl within JuMP.

## Clarabel.jl native interface

To solve the problem directly within Clarabel.jl, we start by creating the solver and settings:
=#

using Clarabel, LinearAlgebra, SparseArrays

settings = Clarabel.Settings(verbose = true)
solver   = Clarabel.Solver()

#=
### Objective function data

We next put the objective function into the standard Clarabel.jl form $\frac{1}{2}x^\top P x + q\top x$.
Define the objective function data as
=#

P = sparse([0. 0.;0. 1.].*2)
q = [0., 0.]
nothing  #hide

#=
### Constraint data

Finally we put the constraints into the standard Clarabel.jl form $Ax + s = b$, where $s \in \mathcal{K}$ for some  cone
$\mathcal{K}$.  We have a single constraint on the 2-norm of a vector, so we rewrite
$$
\left\|\begin{pmatrix} 2x_1 \\ x_2 \end{pmatrix} - \begin{pmatrix} 2 \\ 2 \end{pmatrix}\right\|_2 \le 1
\quad \Longleftrightarrow \quad
\begin{pmatrix} 1 \\ 2x_1 - 2\\ x_2 - 2 \end{pmatrix} \in \mathcal{K}_{SOC}
$$
which puts our constraint in the form $b - Ax \in \mathcal{K}_{SOC}$.  We therefore
define the constraint data as
=#

A = sparse([0.  0.
           -2.  0.;
            0. -1.])
b = [ 1.
     -2.;
     -2.]

cones = [Clarabel.SecondOrderConeT(3)]

nothing  #hide


# Finally we can populate the solver with problem data and solve

Clarabel.setup!(solver, P, q, A, b, cones, settings)
result = Clarabel.solve!(solver)

# then retrieve our solution

result.x


# ## Using JuMP

# We can solve the same problem using
# Clarabel.jl as the backend solver within [JuMP](http://www.juliaopt.org/JuMP.jl/stable/).
# Here is the same problem again:

using Clarabel, JuMP

model = JuMP.Model(Clarabel.Optimizer)
set_optimizer_attribute(model, "verbose", true)

@variable(model, x[1:2])
@constraint(model, [1, 2x[1]-2, x[2] - 2] in SecondOrderCone())
@objective(model, Min, x[2]^2 )

optimize!(model)

# Here is the solution

JuMP.value.(x)

# and the solver termination status again

JuMP.termination_status(model)


# ## Using Convex.jl

# One more time using Clarabel.jl as the backend solver within [Convex.jl](https://jump.dev/Convex.jl/stable/):

using Clarabel, Convex

x = Variable(2)
problem = minimize(square(x[2]))
problem.constraints = [norm([2x[1];x[2]] - [2;2], 2) <= 1]
solve!(problem, Clarabel.Optimizer; silent_solver = false)

# Here is our solution

evaluate(x)

# and the solver termination status again

problem.status
