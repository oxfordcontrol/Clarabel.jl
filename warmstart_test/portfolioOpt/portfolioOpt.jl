using CSV
using Statistics, LinearAlgebra
using SparseArrays
using Clarabel
using TimerOutputs

include("./utils.jl")
include("..//plots.jl")

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

##########################################
# Recording time and iteration number
##########################################
nsample = maxdays - T
solve_time_w = Vector{Float64}(undef,nsample) 
solve_time_c = Vector{Float64}(undef,nsample) 
iter_w = Vector{Int64}(undef,nsample) 
iter_c = Vector{Int64}(undef,nsample) 


#save data needed in a table 
tables = Dict() 
tables[:varVal] = Vector(1:nsample)
tables[:warm_time] = solve_time_w 
tables[:cold_time] = solve_time_c 
tables[:warm_iterations] = iter_w 
tables[:cold_iterations] = iter_c 

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

    if solver.solution.status === Clarabel.SOLVED 
        tables[:warm_time][i] = TimerOutputs.tottime(solver.timers["solve!"])/1e9 
        tables[:warm_iterations][i] = solver.info.iterations 
    else 
        tables[:warm_time][i] = Inf 
    end 

    sleep(0.01)

    #cold start solve
    Clarabel.update_A!(solver_c,A)
    Clarabel.solve!(solver_c)

    if solver_c.solution.status === Clarabel.SOLVED 
        tables[:cold_time][i] = TimerOutputs.tottime(solver_c.timers["solve!"])/1e9 
        tables[:cold_iterations][i] = solver_c.info.iterations 
    else 
        tables[:cold_time][i] = Inf 
    end 
    sleep(0.01) 

end

# ################################################
# #Postprocess 
# ################################################
# tablename = "portfolio"
# filename = ".//..//results//bench_"*tablename*"_table" 
# problems = "Time index"
# save_table(tables,problems,filename,tablename)
# plot_warmstart(iter_w,iter_c,filename)
# geomean = compute_geomean(iter_w,iter_c)
# println("iteration reduction ratio ", 1-geomean_iter, " vs. time reduction ratio ", 1-geomean_time)
# # @save ".//..//results//"*tablename*".jld2" iter_w=iter_w iter_c=iter_c solve_time_w=solve_time_w solve_time_c=solve_time_c varVal=tables[:varVal] 