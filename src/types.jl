using TimerOutputs

# -------------------------------------
# default solver component  types 
# -------------------------------------

# ---------------------
# presolver and internals 
# ---------------------

struct PresolverRowReductionIndex 

    # vector of length = original RHS.   Entries are false
    # for those rows that should be eliminated before solve
    keep_logical::Vector{Bool}

end
struct Presolver{T}

   # original cones of the problem
    init_cones::Vector{SupportedCone}

    # record of reduced constraints for NN cones with inf bounds
    reduce_map::Option{PresolverRowReductionIndex}

    # size of original and reduced RHS, respectively 
    mfull::DefaultInt 
    mreduced::DefaultInt

    # inf bound that was taken from the module level 
    # and should be applied throughout.   Held here so 
    # that any subsequent change to the module's state 
    # won't mess up our solver mid-solve 
    infbound::DefaultFloat 

end

Presolver(args...) = Presolver{DefaultFloat}(args...)

# ---------------
# variables
# ---------------

mutable struct DefaultVariables{T} <: AbstractVariables{T}

    x::AbstractVector{T}
    s::AbstractVector{T}
    z::AbstractVector{T}
    τ::T
    κ::T

    function DefaultVariables{T}(
        n::DefaultInt, 
        m::DefaultInt,
        use_gpu::Bool
    ) where {T}

        x = use_gpu ? CuVector{T}(undef,n) : Vector{T}(undef,n)
        s = use_gpu ? CuVector{T}(undef,m) : Vector{T}(undef,m)
        z = use_gpu ? CuVector{T}(undef,m) : Vector{T}(undef,m)
        τ = T(1)
        κ = T(1)

        new(x,s,z,τ,κ)
    end

end

DefaultVariables(args...) = DefaultVariables{DefaultFloat}(args...)

# Scaling strategy for variables.  Defined
# here to avoid errors due to order of includes

@enum ScalingStrategy begin
    PrimalDual = 0
    Dual       = 1
end


# ---------------
# residuals
# ---------------

mutable struct DefaultResiduals{T} <: AbstractResiduals{T}

    #the main KKT residuals
    rx::AbstractVector{T}
    rz::AbstractVector{T}
    rτ::T

    #partial residuals for infeasibility checks
    rx_inf::AbstractVector{T}
    rz_inf::AbstractVector{T}

    #various inner products.
    #NB: these are invariant w.r.t equilibration
    dot_qx::T
    dot_bz::T
    dot_sz::T
    dot_xPx::T

    #the product Px by itself. Required for infeasibilty checks
    Px::AbstractVector{T}

    function DefaultResiduals{T}(n::DefaultInt,
                                 m::DefaultInt,
                                 use_gpu::Bool) where {T}

        VecType = use_gpu ? CuVector : Vector
        rx = VecType{T}(undef,n)
        rz = VecType{T}(undef,m)
        rτ = T(1)

        rx_inf = VecType{T}(undef,n)
        rz_inf = VecType{T}(undef,m)

        Px = VecType{T}(undef,n)

        new(rx,rz,rτ,rx_inf,rz_inf,zero(T),zero(T),zero(T),zero(T),Px)
    end

end

DefaultResiduals(args...) = DefaultResiduals{DefaultFloat}(args...)


# ---------------
# equilibration data
# ---------------

struct DefaultEquilibration{T} <: AbstractEquilibration{T}

    #scaling matrices for problem data equilibration
    #fields d,e,dinv,einv are vectors of scaling values
    #to be treated as diagonal scaling data
    d::AbstractVector{T}
    dinv::AbstractVector{T}
    e::AbstractVector{T}
    einv::AbstractVector{T}

    #overall scaling for objective function
    c::Base.RefValue{T} 

    function DefaultEquilibration{T}(
        n::DefaultInt,
        m::DefaultInt,
        use_gpu::Bool
    ) where {T}

        c    = Ref(T(1.))

        #Left/Right diagonal scaling for problem data
        d    = (use_gpu) ? CUDA.ones(T,n) : ones(T,n)
        dinv = (use_gpu) ? CUDA.ones(T,n) : ones(T,n)
        e    = (use_gpu) ? CUDA.ones(T,m) : ones(T,m)
        einv = (use_gpu) ? CUDA.ones(T,m) : ones(T,m)

        new(d,dinv,e,einv,c)
    end

end

DefaultEquilibration(args...) = DefaultEquilibration{DefaultFloat}(args...)


# ---------------
# problem data
# ---------------

mutable struct DefaultProblemData{T} <: AbstractProblemData{T}

    P::AbstractMatrix{T}
    q::Vector{T}
    A::AbstractMatrix{T}
    b::Vector{T}
    cones::Vector{SupportedCone}
    n::DefaultInt
    m::DefaultInt
    equilibration::DefaultEquilibration{T}

    # unscaled inf norms of linear terms.  Set to "nothing"
    # during data updating to allow for multiple updates, and 
    # then recalculated during solve if needed

    normq::Option{T}  #unscaled inf norm of q
    normb::Option{T}  #unscaled inf norm of b

    presolver::Option{Presolver{T}}
    dropped_zeroes::DefaultInt #number of eliminated structural zeros

    chordal_info::Option{ChordalInfo{T}}

end

DefaultProblemData(args...) = DefaultProblemData{DefaultFloat}(args...)


mutable struct DefaultProblemDataGPU{T} <: AbstractProblemData{T}

    P::AbstractSparseMatrix{T}
    q::AbstractVector{T}
    A::AbstractSparseMatrix{T}
    At::AbstractSparseMatrix{T}
    b::AbstractVector{T}
    cones::Vector{SupportedCone}
    n::DefaultInt
    m::DefaultInt
    equilibration::DefaultEquilibration{T}

    # unscaled inf norms of linear terms.  Set to "nothing"
    # during data updating to allow for multiple updates, and 
    # then recalculated during solve if needed

    normq::Option{T}  #unscaled inf norm of q
    normb::Option{T}  #unscaled inf norm of b

    presolver::Option{Presolver{T}}
    dropped_zeroes::DefaultInt #number of eliminated structural zeros
    
    chordal_info::Option{ChordalInfo{T}}

end

DefaultProblemDataGPU(args...) = DefaultProblemDataGPU{DefaultFloat}(args...)


# ----------------------
# progress info
# ----------------------

struct LinearSolverInfo 
    name::Symbol
    threads::DefaultInt
    direct::Bool
    nnzA::DefaultInt # nnz in L for A = LDL^T
    nnzL::DefaultInt # nnz in A for A = LDL^T
end

LinearSolverInfo() = LinearSolverInfo(:none, 1, true, 0, 0)


mutable struct DefaultInfo{T} <: AbstractInfo{T}

    mu::T
    sigma::T
    step_length::T
    iterations::UInt32
    cost_primal::T
    cost_dual::T
    res_primal::T
    res_dual::T
    res_primal_inf::T
    res_dual_inf::T
    gap_abs::T
    gap_rel::T
    ktratio::T

    setup_phase_time::Float64
    solve_phase_time::Float64
    solve_time::Float64

    # previous iterate
    prev_cost_primal::T
    prev_cost_dual::T
    prev_res_primal::T
    prev_res_dual::T
    prev_gap_abs::T
    prev_gap_rel::T

    status::SolverStatus

    # linear solver information
    linsolver::LinearSolverInfo

end

function DefaultInfo{T}() where {T}

    #here we set the first set of fields to zero (it doesn't matter),
    #but the previous iterates to Inf to avoid weird edge cases 
    prevvals = ntuple(x->floatmax(T), 6);
    linsolver = LinearSolverInfo()
    DefaultInfo{T}((ntuple(x->0, fieldcount(DefaultInfo)-6-2)...,prevvals...,UNSOLVED,linsolver)...)

end

DefaultInfo(args...) = DefaultInfo{DefaultFloat}(args...)


# ---------------
# solver results
# ---------------

"""
    DefaultSolution{T <: AbstractFloat}
Object returned by the Clarabel solver after calling `optimize!(model)`.

Fieldname | Description
---  | :--- | :---
x | Vector{T}| Primal variable
z | Vector{T}| Dual variable
s | Vector{T}| (Primal) set variable
status | Symbol | Solution status
obj_val | T | Objective value (primal)
obj_val_dual | T | Objective value (dual)
setup_phase_time | T | Solver setup time
solve_phase_time | T | Solver solve time
solve_time | T | Solver total time
iterations | Int | Number of solver iterations
r_prim       | primal residual at termination
r_dual       | dual residual at termination

If the status field indicates that the problem is solved then (x,z,s) are the calculated solution, or a best guess if the solver has terminated early due to time or iterations limits.

If the status indicates either primal or dual infeasibility, then (x,z,s) provide instead an infeasibility certificate.
"""
mutable struct DefaultSolution{T} <: AbstractSolution{T}
    x::AbstractVector{T}
    z::AbstractVector{T}
    s::AbstractVector{T}
    status::SolverStatus
    obj_val::T
    obj_val_dual::T
    setup_phase_time::T 
    solve_phase_time::T
    solve_time::T
    iterations::UInt32
    r_prim::T
    r_dual::T

end

function DefaultSolution{T}(n,m,use_gpu::Bool) where {T <: AbstractFloat}

    VecType = use_gpu ? CuVector : Vector
    x = VecType{T}(undef,n)
    z = VecType{T}(undef,m)
    s = VecType{T}(undef,m)

    # seemingly reasonable defaults
    status  = UNSOLVED
    obj_val = T(NaN)
    obj_val_dual = T(NaN)
    setup_phase_time = zero(T)
    solve_phase_time = zero(T)
    solve_time = zero(T)
    iterations = 0
    r_prim     = T(NaN)
    r_dual     = T(NaN)

  return DefaultSolution{T}(x,z,s,status,obj_val,obj_val_dual,setup_phase_time,solve_phase_time,solve_time,iterations,r_prim,r_dual)
end

DefaultSolution(args...) = DefaultSolution{DefaultFloat}(args...)


# -------------------------------------
# top level solver type
# -------------------------------------

"""
	Solver{T <: AbstractFloat}()
Initializes an empty Clarabel solver that can be filled with problem data using:

    setup!(solver, P, q, A, b, cones, [settings]).

"""
mutable struct Solver{T <: AbstractFloat} <: AbstractSolver{T}

    data::Option{AbstractProblemData{T}}
    variables::Option{AbstractVariables{T}}
    cones::Union{CompositeCone{T},CompositeConeGPU{T},Nothing}
    residuals::Option{AbstractResiduals{T}}
    kktsystem::Option{AbstractKKTSystem{T}}
    info::Option{AbstractInfo{T}}
    step_lhs::Option{AbstractVariables{T}}
    step_rhs::Option{AbstractVariables{T}}
    prev_vars::Option{AbstractVariables{T}}
    solution::Option{AbstractSolution{T}}
    settings::Settings{T}
    timers::TimerOutput

    # all inner constructors initalize most fields to nothing,
    # initialize the timers and copy user settings as needed 

    function Solver{T}(settings::Settings{T}) where {T}
        settings = deepcopy(settings)
        new{T}(ntuple(x->nothing, fieldcount(Solver)-2)...,settings,_timer_init())
    end 

    function Solver{T}() where {T}
        settings = Settings{T}()
        new{T}(ntuple(x->nothing, fieldcount(Solver)-2)...,settings,_timer_init())
    end

    function Solver{T}(d::Dict) where {T}
        settings = Settings{T}(d)
        new{T}(ntuple(x->nothing, fieldcount(Solver)-2)...,settings,_timer_init())
    end

end

Solver(args...; kwargs...) = Solver{DefaultFloat}(args...; kwargs...)

function _timer_init()
    to = TimerOutput()
    #setup the main timer sections here and
    #zero them.   This ensures that the sections
    #exists if we try to clear them later
    @timeit to "setup!" begin (nothing) end
    @timeit to "solve!" begin (nothing) end
    reset_timer!(to["setup!"])
    reset_timer!(to["solve!"])
    to
end 




