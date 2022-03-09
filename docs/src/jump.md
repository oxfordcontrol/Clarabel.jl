# JuMP Interface
Clarabel.jl implements support for [MathOptInterface](https://jump.dev/JuMP.jl/stable/moi/), and is therefore compatible with [JuMP](https://github.com/JuliaOpt/JuMP.jl/).   This allows you to describe and modify your optimisation problem with JuMP and use Clarabel.jl as the backend solver.

## Setting Clarabel.jl Backend

To specify Clarabel.jl as the solver for your JuMP model, load the solver module with `using Clarabel` and then configure Clarabel as the solver backend when initialising the JuMP model:
```julia
model = JuMP.Model(Clarabel.Optimizer)
```

## Solver Settings
Solver-specific settings can be passed after the `Clarabel.Optimizer` object. For example, if you want to adjust the maximum number of iterations and turn off verbose printing use
```julia
set_optimizer_attribute(model, "verbose", true)
set_optimizer_attribute(model, "max_iter", 25)
```
The full list of available settings can be found in the [Settings](@ref api-settings) section of the API Reference.

## Results
After solving the problem the result can be obtained using the standard JuMP commands. To see if the optimisation was successful use
```julia
JuMP.termination_status(model)
JuMP.primal_status(model)
```
If a solution is available, the optimal objective value can be retrieved using
```julia
JuMP.objective_value(model)
```
and the value of a decision variable `x` can be obtained with
```julia
JuMP.value.(x)
```
For more information on JuMP, see the [JuMP documentation](http://www.juliaopt.org/JuMP.jl/stable/).
