using CSV
using Statistics, LinearAlgebra
using SparseArrays, JuMP
using Clarabel
using TimerOutputs
using JLD2

include("./utils.jl")
include("..//plots.jl")

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

# model = Model(Clarabel.Optimizer)
# # set_silent(model)
# @variable(model, η)
# @variable(model, x[1:N])
# @variable(model, w[1:T])
# @variable(model, v[1:T])
# @variable(model, t)

# @constraint(model, sum(x) == 1.0)
# @constraint(model, x .>= 0.0)
# @constraint(model, dot(vec(Er),x) >= r0)
# @constraint(model, w .>= -R*x .- η)
# @constraint(model, w .>= 0.0)
# @constraint(model, [i = 1:T], [v[i], t, w[i]] in MOI.PowerCone(1 / p))
# @constraint(model, sum(v) == t)
# @objective(model, Min, η + inv(1-α)*inv(T^(1/p))*t)

# #We disable the equilibriation for the convenient update of linear costs
# set_optimizer_attribute(model,"equilibrate_enable",false)       
# #We disable the presolve and chordal decomposition for the use of parametric updates
# set_optimizer_attribute(model,"presolve_enable", false)
# set_optimizer_attribute(model,"chordal_decomposition_enable",false)

# optimize!(model)
# @assert is_solved_and_feasible(model)

# solver = model.moi_backend.optimizer.model.optimizer.solver

#solver for the cold_start
solver_c = deepcopy(solver)

##########################################
# Recording time and iteration number
##########################################
nsample = maxdays - T        #Total number of reoptimization
njump = 1          #Interval for a reoptimization
@assert(iszero(T%10))
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

#qp is the linear cost
qnew = deepcopy(solver.data.q)

for i in 1:nsample
    # #Varying value of α
    # global α += 0.01
    # print("α value is ", α)
    # qnew[2] = inv(1-α)*inv(T^(1/p))     #Coefficient w.r.t. variable t

    #Generate new data
    # Rnew = get_return_data(N,(i+1):(i+T)) # return rates of size (T,N)
    # Ernew = mean(Rnew,dims = 1)        #expected return of dim (1,N)
    # Anew = sparse([zeros(1,2) ones(1,N) zeros(1,2*T); 
    #         0 -1 zeros(1,N+T) ones(1,T);
    #         zeros(N+T,2) -I zeros(N+T,T);
    #         zeros(1,2) -Ernew zeros(1,2*T);
    #         -ones(T) zeros(T) -R -I zeros(T,T);
    #         zeros(3*T) kron(ones(T),[0;-1;0]) zeros(3*T,N) kron(Diagonal(ones(T)),[0;0;-1]) kron(Diagonal(ones(T)),[-1;0;0])])
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
    # Clarabel.update_q!(solver,qnew)
    Clarabel.solve!(solver;warmstart=true)

    if solver.solution.status === Clarabel.SOLVED 
        tables[:warm_time][i] = TimerOutputs.tottime(solver.timers["solve!"])/1e9 
        tables[:warm_iterations][i] = solver.info.iterations 
    else
        error("warm start fails at ", i) 
        tables[:warm_time][i] = Inf 
    end 

    sleep(0.1)

    #cold start solve
    Clarabel.update_A!(solver_c,A)
    # Clarabel.update_q!(solver_c,qnew)
    Clarabel.solve!(solver_c)

    if solver_c.solution.status === Clarabel.SOLVED 
        tables[:cold_time][i] = TimerOutputs.tottime(solver_c.timers["solve!"])/1e9 
        tables[:cold_iterations][i] = solver_c.info.iterations 
    else 
        error("cold start fails at ", i) 
        tables[:cold_time][i] = Inf 
    end 
    sleep(0.1) 

end

# ################################################
# #Postprocess 
# ################################################
# tablename = "portfolio_higherorderRisk"
# filename = ".//..//results//bench_"*tablename*"_table" 
# problems = "Times"
# save_table(tables,problems,filename,tablename)
# plot_warmstart(iter_w,iter_c,filename)
# geomean_iter = compute_geomean(iter_w,iter_c)
# geomean_time = compute_geomean(solve_time_w,solve_time_c)
# println("iteration reduction ratio ", 1-geomean_iter, " vs. time reduction ratio ", 1-geomean_time)
# # @save ".//..//results//"*tablename*".jld2" iter_w=iter_w iter_c=iter_c solve_time_w=solve_time_w solve_time_c=solve_time_c varVal=tables[:varVal] 