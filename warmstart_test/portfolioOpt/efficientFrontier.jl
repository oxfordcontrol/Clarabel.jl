using CSV
using Statistics, LinearAlgebra
using SparseArrays
using Clarabel
using TimerOutputs

include("./utils.jl")
include("../plots.jl")

N = 300 # 20 is max. number of assets
T = 500 # num of days used for estimation

R = get_return_data(N,1:T) # return rates of size (T,N)
r = mean(R,dims = 1)        #expected return of dim (1,N)
emd = calc_QR_emd(N,T,R,r)
# r0 = max(0.0,mean(r))
r0 = 0.001
δ = 0.0001

#####################################
#Efficient frontier w.r.t. r0
#####################################
#              min t
#             s.t.  1'*x = 1
#                   (t,U*x) ∈ K_{soc}^{n+1}
#                   x ≥ 0
#                   r'*x ≥ r0
#

#solver setup
P = spzeros(N+1,N+1)
A = sparse([0 ones(1,N);
        -1 zeros(1,N);
        zeros(N) -emd.R;
        zeros(N) -I;
        0 -r])

b = [1;zeros(2*N+1);-r0]
q = vec([1.0;zeros(N)])

cones = [Clarabel.ZeroConeT(1), Clarabel.SecondOrderConeT(N+1), Clarabel.NonnegativeConeT(N+1)]
settings = Clarabel.Settings(equilibrate_enable = true, presolve_enable = false, chordal_decomposition_enable = false)
solver  = Clarabel.Solver()
Clarabel.setup!(solver, P, q, A, b, cones, settings)
Clarabel.solve!(solver)

#solver for the cold_start
solver_c = deepcopy(solver)

bnew = deepcopy(solver.data.b)
##########################################
# Recording time and iteration number
##########################################
nsample = 10
t_values = Vector{Float64}(undef,nsample) 
ft_values = Vector{Float64}(undef,nsample) 
solve_time_w = Vector{Float64}(undef,nsample) 
solve_time_c = Vector{Float64}(undef,nsample) 
iter_w = Vector{Int64}(undef,nsample) 
iter_c = Vector{Int64}(undef,nsample) 


#save data needed in a table 
tables = Dict() 
tables[:t] = t_values
tables[:ft] = ft_values
tables[:warm_time] = solve_time_w 
tables[:cold_time] = solve_time_c 
tables[:warm_iterations] = iter_w 
tables[:cold_iterations] = iter_c 


for i in 1:nsample
    #Increasing r0 gradually
    global r0 += δ 
    println("r0 value is: ", r0)

    #Update the parameter w.r.t. the minimum return 
    bnew[end] = -r0

    #warm start solve
    Clarabel.update_b!(solver,bnew)
    Clarabel.solve!(solver;warmstart=true)

    if solver.solution.status === Clarabel.SOLVED 
        tables[:warm_time][i] = TimerOutputs.tottime(solver.timers["solve!"])/1e9 
        tables[:warm_iterations][i] = solver.info.iterations 
    else 
        tables[:warm_time][i] = Inf 
    end 

    sleep(0.01)

    #cold start solve
    Clarabel.update_b!(solver_c,bnew)
    Clarabel.solve!(solver_c)

    if solver_c.solution.status === Clarabel.SOLVED 
        tables[:cold_time][i] = TimerOutputs.tottime(solver_c.timers["solve!"])/1e9 
        tables[:cold_iterations][i] = solver_c.info.iterations 
    else 
        tables[:cold_time][i] = Inf 
    end 
    sleep(0.01) 

    tables[:t][i] = r0
    tables[:ft][i] = solver.solution.obj_val

end

################################################
#Postprocess 
################################################
filename = ".//..//results//bench_portfolio_efficientFrontier_table" 
problems = "\$ r_0 \$ value"
save_efficientFrontier_table(tables,problems,filename)
plot_warmstart(iter_w,iter_c,filename)
geomean_iter = compute_geomean(iter_w,iter_c)
geomean_time = compute_geomean(solve_time_w,solve_time_c)