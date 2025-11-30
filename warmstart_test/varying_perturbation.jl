using Random, StatsBase, LinearAlgebra
using SparseArrays
using Revise
using Clarabel, JuMP
using ClarabelBenchmarks
using TimerOutputs

include("save_table.jl")

class   = "mpc"

problems = keys(ClarabelBenchmarks.PROBLEMS[class])

nsample = length(problems)
solve_time_w = Vector{Float64}(undef,nsample)
solve_time_c = Vector{Float64}(undef,nsample)
iter_w = Vector{Int64}(undef,nsample)
iter_c = Vector{Int64}(undef,nsample)
ratio_time = Vector{Float64}(undef,nsample)
ratio_iter = Vector{Float64}(undef,nsample)

#save data needed in a table
tables = Dict()
tables[:problems] = keys(ClarabelBenchmarks.PROBLEMS[class])
tables[:varVal] = Vector(1:nsample)
tables[:warm_time] = solve_time_w 
tables[:cold_time] = solve_time_c 
tables[:warm_iterations] = iter_w 
tables[:cold_iterations] = iter_c 
tables[:ratio_time] = ratio_time
tables[:ratio_iter] = ratio_iter

lenr = 31
geomean_iter = Vector{Float64}(undef,lenr)
geomean_time = Vector{Float64}(undef,lenr)

###################################################
# YC: We can switch among perturbation on q,b, and A
# Now, code is for the update of A
###################################################
#Varying noise level
for (k,noise) in enumerate(10.0.^(range(-3,0,length=lenr)))
    for (i,example) in enumerate(problems)

        # test examples
        println("current problem ", i, " is ", example)

        #Initialize a solver
        model = Model(Clarabel.Optimizer)
        # set_optimizer_attribute(model,"tol_gap_abs", 1e-6)
        # set_optimizer_attribute(model,"tol_gap_rel", 1e-6)
        # set_optimizer_attribute(model,"tol_feas", 1e-6)
        # set_optimizer_attribute(model,"tol_ktratio", 1e-4)
        #We disable the equilibriation for the convenient update of linear costs
        set_optimizer_attribute(model,"equilibrate_enable",false)       
        #We disable the presolve and chordal decomposition for the use of parametric updates
        set_optimizer_attribute(model,"presolve_enable", false)
        set_optimizer_attribute(model,"chordal_decomposition_enable",false)
        ClarabelBenchmarks.PROBLEMS[class][example](model)
        solver = model.moi_backend.optimizer.model.optimizer.solver
        solver_c = deepcopy(solver)

        #generating perturbation
        seed = Random.MersenneTwister(1000 + i)

        # Generate the samples
        # q = solver.data.q
        # len = length(q)
        # num_noise = Int(min(ceil(0.1*len),20))
        # println("num noise: ", num_noise)

        # indices = StatsBase.sample(seed,1:len, num_noise; replace = false)
        # noiseq = (1 .- 2*rand(seed, num_noise))

        # #Perturb only nonzero entries
        # qnew = deepcopy(q)

        # for (j,ele) in enumerate(indices)
        #     if abs(q[ele]) > 1e-6
        #         qnew[ele] = (1 + noise*noiseq[j])*q[ele]
        #     else
        #         qnew[ele] = noise*noiseq[j]
        #     end
        # end
        A = solver.data.A
        len = length(A.nzval)
        num_noise = Int(min(ceil(0.1*len),20))
        @assert(length(A.nzval) > 20)
        println("num noise: ", num_noise)

        indices = StatsBase.sample(seed,1:len, num_noise; replace = false)
        noiseV = (1 .- 2*rand(seed, num_noise))

        #Perturb only nonzero entries
        Anew = deepcopy(A)

        for (j,ele) in enumerate(indices)
            if abs(A.nzval[ele]) > 1e-6
                Anew.nzval[ele] = (1 + noise*noiseV[j])*Anew.nzval[ele]
            else
                Anew.nzval[ele] = noise*noiseV[j]
            end
        end

        #warm start solve
        Clarabel.update_A!(solver,Anew)
        # Clarabel.update_b!(solver,bnew)
        # Clarabel.update_q!(solver,qnew)
        Clarabel.solve!(solver;warmstart=true)

        if solver.solution.status === Clarabel.SOLVED 
            tables[:warm_time][i] = TimerOutputs.tottime(solver.timers["solve!"])/1e9 
            tables[:warm_iterations][i] = solver.info.iterations 
        else 
            tables[:warm_time][i] = Inf 
        end 

        sleep(0.01)

        #cold start solve
        Clarabel.update_A!(solver_c,Anew)
        # Clarabel.update_b!(solver_c,bnew)
        # Clarabel.update_q!(solver_c,qnew)
        Clarabel.solve!(solver_c)

        if solver_c.solution.status === Clarabel.SOLVED 
            tables[:cold_time][i] = TimerOutputs.tottime(solver_c.timers["solve!"])/1e9 
            tables[:cold_iterations][i] = solver_c.info.iterations 
        else 
            tables[:cold_time][i] = Inf 
        end 
        sleep(0.01) 

        #record reduction ratios
        if solver_c.solution.status === Clarabel.SOLVED && solver.solution.status === Clarabel.SOLVED 
            tables[:ratio_iter][i] = tables[:warm_iterations][i]/tables[:cold_iterations][i]
            tables[:ratio_time][i] = tables[:warm_time][i]/tables[:cold_time][i]
        else
            tables[:ratio_iter][i] = Inf
            tables[:ratio_time][i] = Inf
        end
    end

    # Create a boolean mask where both a and b are finite
    finite_mask = isfinite.(tables[:ratio_iter]) .&& tables[:ratio_iter] .> 0.0

    # Use the mask to filter both vectors
    geomean_iter[k] = exp(mean(log.(tables[:ratio_iter][finite_mask])))
    geomean_time[k] = exp(mean(log.(tables[:ratio_time][finite_mask])))
end


# ################################################
# #Postprocess 
# ################################################

@save ".//results//"*class*"_varying_disturbance_A.jld2" geomean_iter=geomean_iter geomean_time=geomean_time

geomean_iter_A = JLD2.jldopen(".//results//mpc_varying_disturbance_A.jld2", "r") do file
    file["geomean_iter"]
end
geomean_iter_b = JLD2.jldopen(".//results//mpc_varying_disturbance_b.jld2", "r") do file
    file["geomean_iter"]
end
geomean_iter_q = JLD2.jldopen(".//results//mpc_varying_disturbance_q.jld2", "r") do file
    file["geomean_iter"]
end