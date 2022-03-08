The source files for all examples can be found in [/examples](https://github.com/oxfordcontrol/COSMO.jl/tree/master/examples/).
```@meta
EditURL = "<unknown>/.julia/dev/Clarabel/examples/example.jl"
```

# Simple QP Example

````@example example
#Required packages
using LinearAlgebra, SparseArrays
using Clarabel
````

````@example example
#Problem data in sparse format
A = SparseMatrixCSC(I(3)*1.)
P = SparseMatrixCSC(I(3)*1.)
A = [A;-A]
c = [3.;-2.;1.]*10
b = ones(Float64,2*3);
nothing #hide
````

----------------------------
### Solve in Clarabel native interface

````@example example
cone_types = [Clarabel.NonnegativeConeT]
cone_dims  = [length(b)]

settings = Clarabel.Settings(
        max_iter=20,
        verbose=false,
        direct_kkt_solver=true,
        equilibrate_enable = true
)
solver = Clarabel.Solver()
Clarabel.setup!(solver,P,c,A,b,cone_types,cone_dims,settings)
Clarabel.solve!(solver)
x = solver.variables.x
````

-------------
### Solve in JuMP

````@example example
using JuMP
model = Model(Clarabel.Optimizer)
@variable(model, x[1:3])
@constraint(model, c1, A*x .<= b)
@objective(model, Min, sum(c.*x) + 1/2*x'*P*x)

#Run the opimization
optimize!(model)
x = JuMP.value.(x)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

