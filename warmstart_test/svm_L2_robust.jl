using MLDatasets
using Clarabel
using JuMP
using JLD2
using TimerOutputs

include("save_table.jl")

trainset = MNIST(:train)
length(trainset)
X_train, Y_train = trainset[:]
(a,b,c) = size(X_train)
X_train = reshape(X_train,(a*b,c))

#take a subset of points
njump = 2
X = X_train[:,1:njump:end]
Y = Y_train[1:njump:end]
#Suppose we are going to train a binary classifier for the number 1
Y = map(x -> x == 1 ? 1 : -1, Y)

#n is the feature number and m is the number of data points
(n,m) = size(X)

λ = 0.01

model = Model(Clarabel.Optimizer)
@variable(model, β[1:n])
@variable(model, v)
@variable(model, α)
@variable(model, t[1:m])
@constraint(model, 1.0 .- Y.*(X'*β .- v) .<= t)
@constraint(model, t .>= 0)
@constraint(model, [α; β] in MOI.SecondOrderCone(length(β) + 1))
@objective(model, Min, sum(t)/m + λ*α)

#We disable the equilibriation for the convenient update of linear costs
set_optimizer_attribute(model,"equilibrate_enable",false)
#We disable the presolve and chordal decomposition for the use of parametric updates
set_optimizer_attribute(model,"presolve_enable", false)
set_optimizer_attribute(model,"chordal_decomposition_enable",false)

optimize!(model)
solver = JuMP.unsafe_backend(model).solver     #warm start
solver_c = deepcopy(solver)                                     #cold start
solver_cc = deepcopy(solver)  # independent convex-combination sequence
warmup_comparison!(solver; weight=0.01)
#Clarabel.solve!(solver)

#The linear cost vector q where the first entry is λ
qs = deepcopy(solver.data.q)
λ_index = only(findall(==(λ),qs))

#################################################
# Tuning hyperparameter λ
#################################################
nsample = 10
step = 0.01
rng = range(λ+step,stop = nsample*step+λ, length = nsample)


##########################################
# Recording time and iteration number
##########################################
solve_time_w = fill(Inf,nsample)
solve_time_c = fill(Inf,nsample)
iter_w = fill(-1,nsample)
iter_c = fill(-1,nsample)


#save data needed in a table
tables = Dict()
tables[:varVal] = Vector(rng)
tables[:warm_time] = solve_time_w
tables[:cold_time] = solve_time_c
tables[:warm_iterations] = iter_w
tables[:cold_iterations] = iter_c
init_convex_comparison!(tables,nsample; weight=0.01)
tables[:output_prefix] = joinpath(@__DIR__,"results","svm_robust_hyperparameter_tuning")
iter_cc = tables[:convex_iterations]
solve_time_cc = tables[:convex_time]

#warm start
for (i,λv) in enumerate(tables[:varVal])
    qs[λ_index] = λv
    Clarabel.update_q!(solver,qs)
    Clarabel.solve!(solver;warmstart=true)

    record_comparison_result!(tables,:warm,solver,i)

    # Convex combination of its own previous solution and the cone unit start.
    Clarabel.update_q!(solver_cc,qs)
    solve_convex_comparison!(solver_cc,tables,i)
    sleep(0.1)

    Clarabel.update_q!(solver_c,qs)
    Clarabel.solve!(solver_c)

    record_comparison_result!(tables,:cold,solver_c,i)
    sleep(0.1)
end

################################################
#Postprocess
################################################
tablename = "svm_robust_hyperparameter_tuning"
filename = joinpath(@__DIR__,"results","bench_"*tablename*"_comparison_table")
problems = "\$ \\lambda \$ value"
save_table(tables,problems,filename,tablename)
geomean_iter = compute_geomean(iter_w,iter_c)
geomean_time = compute_geomean(solve_time_w,solve_time_c)
geomean_iter_cc = compute_geomean(iter_cc,iter_c)
geomean_time_cc = compute_geomean(solve_time_cc,solve_time_c)

save_comparison(tables,joinpath(@__DIR__,"results",tablename))
