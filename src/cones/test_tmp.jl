using JuMP, Mosek, MosekTools


#compute dualgradient
model = Model(Mosek.Optimizer)
@variable(model, x[1:3])
@variable(model, z[1:3])
@variable(model, t)
s = [10.; 0; -2]
@objective(model, Min, x'*s - z[1] - z[2] - z[3])
@constraint(model, [t, x[2], x[1]] in MOI.ExponentialCone())
@constraint(model, [z[1], 1., t - x[3]] in MOI.ExponentialCone())
@constraint(model, [z[2], 1., x[1]] in MOI.ExponentialCone())
@constraint(model, [z[3], 1., x[2]] in MOI.ExponentialCone())

optimize!(model)

println(value.(x))
