using CSV
using Statistics, LinearAlgebra
using SparseArrays
using Clarabel
using JLD2
using TimerOutputs

include("./utils.jl")

N = 300 # 20 is max. number of assets
T = 500 # num of days used for estimation
maxdays = T+100 #maximum days

R = get_return_data(N,1:T) # return rates of size (T,N)
r = mean(R,dims = 1)        #expected return of dim (1,N)
VarM = calc_variance(N,T,R,r)
emd = cholesky(VarM)
r0 = mean(r)

#####################################
#Portfolio problem
#####################################
#              min t
#             s.t.  1'*x = 1
#                   (t,U*x) ∈ K_{soc}^{n+1}
#                   x ≥ 0
#                   r'*x ≥ r0
#

Rloc = emd.U*1e8
rloc = ones(1,N)
#solver setup
P = spzeros(N+1,N+1)
A = sparse([0 ones(1,N);
        -1 zeros(1,N);
        zeros(N) Rloc;
        zeros(N) -I;
        0 rloc])

#Memory reusage later on
AR = @view A[3:N+2,2:N+1]
Ar = @view A[2*N+3:2*N+3,2:N+1]
AR .= -emd.U
Ar .= -r

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

##########################################
# Recording time and iteration number
##########################################
nsample = maxdays - T
solve_time_w = fill(Inf,nsample)
solve_time_c = fill(Inf,nsample)
iter_w = fill(-1,nsample)
iter_c = fill(-1,nsample)


#save data needed in a table
tables = Dict()
tables[:varVal] = Vector(1:nsample)
tables[:warm_time] = solve_time_w
tables[:cold_time] = solve_time_c
tables[:warm_iterations] = iter_w
tables[:cold_iterations] = iter_c
init_convex_comparison!(tables,nsample; weight=0.01)
tables[:output_prefix] = joinpath(@__DIR__,"..","results","portfolio")
iter_cc = tables[:convex_iterations]
solve_time_cc = tables[:convex_time]

njump = 1

for i in 1:nsample
    println("Current problem is ", i)
    Rnew = get_return_data(N,(i*njump+1):(i*njump+T))  # return rates of size (T,N)
    rnew = mean(Rnew,dims = 1)        #expected return of dim (1,N)
    VarM = calc_variance(N,T,Rnew,rnew)
    emdnew = cholesky(VarM)

    global AR .= -emdnew.U
    global Ar .= -rnew

    #warm start solve
    Clarabel.update_A!(solver,A)
    Clarabel.solve!(solver;warmstart=true)

    record_comparison_result!(tables,:warm,solver,i)

    # Convex combination of its own previous solution and the cone unit start.
    Clarabel.update_A!(solver_cc,A)
    solve_convex_comparison!(solver_cc,tables,i)

    sleep(0.01)

    #cold start solve
    Clarabel.update_A!(solver_c,A)
    Clarabel.solve!(solver_c)

    record_comparison_result!(tables,:cold,solver_c,i)
    sleep(0.01)

end

# Save all three methods separately from historical results.
filename = joinpath(@__DIR__,"..","results","bench_portfolio_comparison_table")
save_table(tables,"Time index",filename,"portfolio_comparison")
save_comparison(tables,joinpath(@__DIR__,"..","results","portfolio"))
geomean_iter = compute_geomean(iter_w,iter_c)
geomean_time = compute_geomean(solve_time_w,solve_time_c)
geomean_iter_cc = compute_geomean(iter_cc,iter_c)
geomean_time_cc = compute_geomean(solve_time_cc,solve_time_c)
