using CSV
using Statistics, LinearAlgebra
using SparseArrays
using Clarabel
using JLD2
using TimerOutputs

include("./utils.jl")

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
solver_cc = deepcopy(solver)  # independent convex-combination sequence
warmup_comparison!(solver; weight=0.01)

bnew = deepcopy(solver.data.b)
##########################################
# Recording time and iteration number
##########################################
nsample = 10
t_values = fill(Inf,nsample)
ft_values = fill(Inf,nsample)
solve_time_w = fill(Inf,nsample)
solve_time_c = fill(Inf,nsample)
iter_w = fill(-1,nsample)
iter_c = fill(-1,nsample)


#save data needed in a table
tables = Dict()
tables[:t] = t_values
tables[:ft] = ft_values
tables[:warm_time] = solve_time_w
tables[:cold_time] = solve_time_c
tables[:warm_iterations] = iter_w
tables[:cold_iterations] = iter_c
init_convex_comparison!(tables,nsample; weight=0.01)
tables[:output_prefix] = joinpath(@__DIR__,"..","results","portfolio_efficientFrontier")
iter_cc = tables[:convex_iterations]
solve_time_cc = tables[:convex_time]


for i in 1:nsample
    #Increasing r0 gradually
    global r0 += δ
    println("r0 value is: ", r0)

    #Update the parameter w.r.t. the minimum return
    bnew[end] = -r0

    #warm start solve
    Clarabel.update_b!(solver,bnew)
    Clarabel.solve!(solver;warmstart=true)

    record_comparison_result!(tables,:warm,solver,i)

    # Convex combination of its own previous solution and the cone unit start.
    Clarabel.update_b!(solver_cc,bnew)
    solve_convex_comparison!(solver_cc,tables,i)

    sleep(0.01)

    #cold start solve
    Clarabel.update_b!(solver_c,bnew)
    Clarabel.solve!(solver_c)

    record_comparison_result!(tables,:cold,solver_c,i)
    sleep(0.01)

    tables[:t][i] = r0
    tables[:ft][i] = solver.solution.obj_val

end

################################################
#Postprocess
################################################
filename = joinpath(@__DIR__,"..","results","bench_portfolio_efficientFrontier_comparison_table")
problems = "\$ r_0 \$ value"
save_efficientFrontier_table(tables,problems,filename)
geomean_iter = compute_geomean(iter_w,iter_c)
geomean_time = compute_geomean(solve_time_w,solve_time_c)
geomean_iter_cc = compute_geomean(iter_cc,iter_c)
geomean_time_cc = compute_geomean(solve_time_cc,solve_time_c)
save_comparison(tables,joinpath(@__DIR__,"..","results","portfolio_efficientFrontier"))
