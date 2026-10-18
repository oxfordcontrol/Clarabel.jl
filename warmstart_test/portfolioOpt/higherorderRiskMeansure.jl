using CSV
using Statistics, LinearAlgebra
using SparseArrays
using Clarabel
using TimerOutputs
using JLD2

include("./utils.jl")

N = 300 # 20 is max. number of assets
T = 500 # num of days used for estimation
maxdays = T + 100 #maximum days

R = get_return_data(N,1:T) # return rates of size (T,N)
Er = mean(R,dims = 1)        #expected return of dim (1,N)

#Initial volume
α = 0.99
p = 3
r0 = mean(Er)

###################################################
# Portfolio problem with higher-order risk measure,
# from the paper https://doi.org/10.1080/14697680701458307
###################################################
#              min η + t/((1-α)*T^{1/p})
#             s.t.  1'*x = 1
#                   E(r)'*x ≥ r0
#                   w .>= -R*x .- η
#                   (t,w)  in the p_norm cone
#                   x ≥ 0
#                   w_j \ge 0, j = 1,...,J
#


#solver setup
P = spzeros(N+2*T+2,N+2*T+2)
A = sparse([zeros(1,2) ones(1,N) zeros(1,2*T);
            0 -1 zeros(1,N+T) ones(1,T);
            zeros(N+T,2) -I zeros(N+T,T);
            zeros(1,2) rand(1,N) zeros(1,2*T);
            -ones(T) zeros(T) rand(T,N) -I zeros(T,T);
            zeros(3*T) kron(ones(T),[0;-1;0]) zeros(3*T,N) kron(Diagonal(ones(T)),[0;0;-1]) kron(Diagonal(ones(T)),[-1;0;0])])

#Initialization for entries corresponding to the expected return Er and R matrix
Aex = @view A[N+T+3:N+T+3,3:N+2]
AR = @view A[N+T+4:N+T+3+T,3:N+2]
Aex .= -Er
AR .= -R

b = [1;0;zeros(N+T); -r0; zeros(T);zeros(3*T)]
q = vec([1;inv(1-α)*inv(T^(1/p));zeros(N+2*T)])

cones  = sizehint!(Clarabel.SupportedCone[],T+2)
push!(cones,Clarabel.ZeroConeT(2))
push!(cones,Clarabel.NonnegativeConeT(N+2*T+1))

for i in 1:T
    push!(cones,Clarabel.PowerConeT(1/p))
end

settings = Clarabel.Settings(equilibrate_enable = true, presolve_enable = false, chordal_decomposition_enable = false,
                            static_regularization_constant = 1e-7,
                            tol_gap_abs = 1e-7, tol_gap_rel = 1e-7, tol_feas = 1e-7, tol_ktratio = 1e-5)
solver  = Clarabel.Solver()
Clarabel.setup!(solver, P, q, A, b, cones, settings)
Clarabel.solve!(solver)

#solver for the cold_start
solver_c = deepcopy(solver)
# Compile smoothing on a disposable copy before recording solve times.
let scratch = deepcopy(solver)
    scratch.settings.verbose = false
    Clarabel.solve!(scratch; warmstart=true)
end

##########################################
# Recording time and iteration number
##########################################
nsample = maxdays - T        #Total number of reoptimization
njump = 1          #Interval for a reoptimization
@assert(iszero(T%10))
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
for i in 1:nsample
    println("current example is ", i)
    #Update in a loop
    loop_i = ((i-1)*njump + 1)%T
    tables[:varVal][i] = i

    Rnew = get_return_data(N,(i-1)*njump + 1+T:(i*njump+T)) # return rates of size (T,N)
    @views R[loop_i:loop_i+njump-1,:] .= Rnew
    Ernew = mean(R,dims = 1)        #expected return of dim (1,N)

    global Aex .= -Ernew
    global @views AR[loop_i:loop_i+njump-1,:] .= -Rnew

    #warm start solve
    Clarabel.update_A!(solver,A)
    Clarabel.solve!(solver;warmstart=true)

    solver.solution.status == Clarabel.SOLVED || error("Smoothing failed at update $i")
    solve_time_w[i] = TimerOutputs.tottime(solver.timers["solve!"])/1e9
    iter_w[i] = solver.info.iterations

    #cold start solve
    Clarabel.update_A!(solver_c,A)
    Clarabel.solve!(solver_c)

    solver_c.solution.status == Clarabel.SOLVED || error("Cold start failed at update $i")
    solve_time_c[i] = TimerOutputs.tottime(solver_c.timers["solve!"])/1e9
    iter_c[i] = solver.info.iterations

end

# Preserve the original HMCR data used in the draft when rerunning this script.
output = joinpath(@__DIR__,"..","results","portfolio_higherorderRisk_rerun.jld2")
JLD2.jldsave(output; iter_w,iter_c,solve_time_w,solve_time_c,varVal=tables[:varVal],
            geomean_iter=compute_geomean(iter_w,iter_c),
            geomean_time=compute_geomean(solve_time_w,solve_time_c))
println("Saved two-method HMCR results to ",output)
