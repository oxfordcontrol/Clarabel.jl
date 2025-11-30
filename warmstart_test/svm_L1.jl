using MLDatasets
using Clarabel
using JuMP
using TimerOutputs

include("save_table.jl")
include("plots.jl")

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

init_λ = 0.01
λ = init_λ

model = Model(Clarabel.Optimizer)
@variable(model, β[1:n])
@variable(model, v)
@variable(model, α)
@variable(model, t[1:m])
@constraint(model, 1.0 .- Y.*(X'*β .- v) .<= t)
@constraint(model, t .>= 0)
@constraint(model, [α; β] in MOI.NormOneCone(length(β) + 1))
@objective(model, Min, sum(t)/m + λ*α)

#We disable the equilibriation for the convenient update of linear costs
set_optimizer_attribute(model,"equilibrate_enable",false)       
#We disable the presolve and chordal decomposition for the use of parametric updates
set_optimizer_attribute(model,"presolve_enable", false)
set_optimizer_attribute(model,"chordal_decomposition_enable",false)

optimize!(model)
solver = model.moi_backend.optimizer.model.optimizer.solver     #warm start
solver_c = deepcopy(solver)                                     #cold start
#Clarabel.solve!(solver)

#The linear cost vector q where the first entry is λ 
qs = deepcopy(solver.data.q)

#################################################
# Tuning hyperparameter λ
#################################################
nsample = 10
rng = range(2*λ,stop = (nsample+1)*λ, length = nsample)


##########################################
# Recording time and iteration number
##########################################
solve_time_w = Vector{Float64}(undef,nsample) 
solve_time_c = Vector{Float64}(undef,nsample) 
iter_w = Vector{Int64}(undef,nsample) 
iter_c = Vector{Int64}(undef,nsample) 


#save data needed in a table 
tables = Dict() 
tables[:varVal] = Vector(rng)
tables[:warm_time] = solve_time_w 
tables[:cold_time] = solve_time_c 
tables[:warm_iterations] = iter_w 
tables[:cold_iterations] = iter_c 

@assert(qs[786] == init_λ)
#warm start
for (i,λv) in enumerate(tables[:varVal]) 
    qs[786] = λv
    Clarabel.update_q!(solver,qs)
    Clarabel.solve!(solver;warmstart=true)

    if solver.solution.status === Clarabel.SOLVED 
        tables[:warm_time][i] = TimerOutputs.tottime(solver.timers["solve!"])/1e9 
        tables[:warm_iterations][i] = solver.info.iterations 
    else 
        tables[:warm_time][i] = Inf 
    end 
    sleep(0.1) 

    Clarabel.update_q!(solver_c,qs)
    Clarabel.solve!(solver_c)

    if solver_c.solution.status === Clarabel.SOLVED 
        tables[:cold_time][i] = TimerOutputs.tottime(solver_c.timers["solve!"])/1e9 
        tables[:cold_iterations][i] = solver_c.info.iterations 
    else 
        tables[:cold_time][i] = Inf 
    end 
    sleep(0.1) 
end

################################################
#Postprocess 
################################################
tablename = "svm_hyperparameter_tuning"
filename = ".//results//bench_"*tablename*"_table" 
problems = "\$ \\lambda \$ value"
save_table(tables,problems,filename,tablename)
geomean_iter = compute_geomean(iter_w,iter_c)
geomean_time = compute_geomean(solve_time_w,solve_time_c)

@save ".//results//"*tablename*".jld2" iter_w=iter_w iter_c=iter_c solve_time_w=solve_time_w solve_time_c=solve_time_c varVal=tables[:varVal] 