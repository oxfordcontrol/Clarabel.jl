#=
# Basic QP Example

Suppose that we want to solve the following 2-dimensional quadratic programming problem:

$$
\begin{array}{ll} \text{minimize} &  3x_1^2 + 2x_2^2 - x_1 - 4x_2\\
\text{subject to} &  -1 \leq x \leq 1, ~ x_1 = 2x_2
\end{array}
$$

In this example we will see how to solve this problem both natively in Clarabel.jl
and also by solving with Clarabel.jl within either JuMP or Convex.jl.

## Clarabel.jl native interface

To solve the problem directly within Clarabel.jl, we start by creating the solver and settings:
=#

using Clarabel, LinearAlgebra, SparseArrays

settings = Clarabel.Settings(verbose = true)
solver   = Clarabel.Solver()

#=
### Objective function data

We next put the objective function into the standard Clarabel.jl form $\frac{1}{2}x^\top P x + q^\top x$.
Define the objective function data as
$P = 2 \cdot \begin{bmatrix} 3 & 0 \\ 0 & 2\end{bmatrix}$ and
$q = \begin{bmatrix} -1 \\ -4\end{bmatrix}$.
=#

P = sparse([3. 0.;0. 2.].*2)
q = [-1., -4.]
nothing  #hide

#=
### Constraint data

Finally we put the constraints into the standard Clarabel.jl form $Ax + s = b$, where $s \in \mathcal{K}$ for some
composite cone $\mathcal{K}$.   We have 1 equality constraint and 6 inequalities, so we require the first element of $s$
to be zero (i.e. the first constraint will correspond to the equality) and all other elements $s_i \ge 0$.   Our
condition on $s$ is therefore $s \in K = \{0\}^1 \times \mathbb{R}^4_{\ge 0}.$ Define the constraint data as $A =
\footnotesize\begin{bmatrix*}[r] 1 & -2 \\ 1 & 0 \\ 0 & 1 \\ -1 & 0 \\ 0 & -1\end{bmatrix*}$ and $b=
\footnotesize\begin{bmatrix} 0 \\ 1 \\ 1 \\ 1 \\ 1 \end{bmatrix}$.

=#

A = sparse([1. -2.;    #<-- LHS of equality constraint
            1.  0.;    #<-- LHS of inequality constraint (upper bound)
            0.  1.;    #<-- LHS of inequality constraint (upper bound)
           -1.  0.;    #<-- LHS of inequality constraint (lower bound)
            0. -1.;    #<-- LHS of inequality constraint (lower bound)
            ])

b = [0.;        #<-- RHS of equality constraint
     ones(4)   #<-- RHS of inequality constraints
    ]

cones =
    [Clarabel.ZeroConeT(1),           #<--- for the equality constraint
     Clarabel.NonnegativeConeT(4)]    #<--- for the inequality constraints


nothing  #hide

# Finally we can populate the solver with problem data and solve

Clarabel.setup!(solver, P, q, A, b, cones, settings)
result = Clarabel.solve!(solver)

# then retrieve our solution

result.x

#=
!!! tip
    There is no constraint on the ordering of the cones that appears
    in `cones` as long as it is compatible with the ordering of
    the constraints as they appear in `A` and `b`.   There is also no
    constraint on the number of instances of each type that appear.
    You could, for example, define the inequalities in the above example
    using


    ```julia
    cones =
        [Clarabel.ZeroConeT(1),           #<--- for the equality constraint
         Clarabel.NonnegativeConeT(2),    #<--- first half of the inequality constraints
         Clarabel.NonnegativeConeT(2)]    #<--- second half of the inequality constraints

    and get the same result.
=#


# ## Using JuMP

# We can solve the same problem a little more easily by using
# Clarabel.jl as the backend solver within [JuMP](http://www.juliaopt.org/JuMP.jl/stable/).
# Here is the same problem again:

using Clarabel, JuMP

model = JuMP.Model(Clarabel.Optimizer)
set_optimizer_attribute(model, "verbose", true)

@variable(model, x[1:2])
@constraint(model, x[1] == 2x[2])
@constraint(model,  -1 .<= x .<= 1)
@objective(model, Min, 3x[1]^2 + 2x[2]^2 - x[1] - 4x[2])

optimize!(model)

# Here is the solution

JuMP.value.(x)

# and the solver termination status again

JuMP.termination_status(model)


# ## Using Convex.jl

# We can likewise solve the same problem a using
# Clarabel.jl as the backend solver within [Convex.jl](https://jump.dev/Convex.jl/stable/).   Here is the same problem one more time:

using Clarabel, Convex

x = Variable(2)
problem = minimize(3square(x[1]) + 2square(x[2]) - x[1] - 4x[2])
problem.constraints = [x[1] == 2x[2]]
problem.constraints += [x >= -1; x <= 1]
solve!(problem, Clarabel.Optimizer; silent_solver = false)

# Here is our solution

evaluate(x)

# and the solver termination status again

problem.status

#=
!!! warning
    Note that in the Clarabel.jl output that follows the call to solve! using Convex.jl, the problem posed
    to the solver has been converted to a second-order cone program with a linear objective.   You can see this
    because now `nnz(P) == 0` (there is no quadratic term in the objective) and the solver reports two second order
    cone constraints.

     Although the solution will be the same, the required number of iterations and solve time are slightly higher.
     When solving problems with quadratic objectives in Clarabel.jl, it is generally preferable to use either the
     native Clarabel.jl interface or JuMP, both of which handle quadratic terms in the objective directly.

=#
