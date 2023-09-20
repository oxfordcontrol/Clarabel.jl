using TimerOutputs

# -------------------------------------
# abstract type defs
# -------------------------------------
abstract type AbstractVariables{T <: AbstractFloat}   end
abstract type AbstractEquilibration{T <: AbstractFloat}   end
abstract type AbstractResiduals{T <: AbstractFloat}   end
abstract type AbstractProblemData{T <: AbstractFloat} end
abstract type AbstractKKTSystem{T <: AbstractFloat} end
abstract type AbstractKKTSolver{T <: AbstractFloat} end
abstract type AbstractInfo{T <: AbstractFloat} end
abstract type AbstractSolution{T <: AbstractFloat} end
abstract type AbstractSolver{T <: AbstractFloat}   end

# -------------------------------------
# default solver subcomponent implementations
# -------------------------------------

# ---------------
# variables
# ---------------

mutable struct DefaultVariables{T} <: AbstractVariables{T}

    x::Vector{T}
    s::ConicVector{T}
    z::ConicVector{T}
    τ::T
    κ::T

    function DefaultVariables{T}(
        n::Integer, 
        cones::CompositeCone{T}
    ) where {T}

        x = Vector{T}(undef,n)
        s = ConicVector{T}(cones)
        z = ConicVector{T}(cones)
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
# equilibration data
# ---------------

struct DefaultEquilibration{T} <: AbstractEquilibration{T}

    #scaling matrices for problem data equilibration
    #fields d,e,dinv,einv are vectors of scaling values
    #to be treated as diagonal scaling data
    d::Vector{T}
    dinv::Vector{T}
    e::ConicVector{T}
    einv::ConicVector{T}

    #overall scaling for objective function
    c::Base.RefValue{T}

    function DefaultEquilibration{T}(
        nvars::Int,
        cones::CompositeCone{T},
    ) where {T}

        #Left/Right diagonal scaling for problem data
        d    = ones(T,nvars)
        dinv = ones(T,nvars)
        e    = ConicVector{T}(cones); e    .= one(T)
        einv = ConicVector{T}(cones); einv .= one(T)

        c    = Ref(T(1.))

        new(d,dinv,e,einv,c)
    end

end

DefaultEquilibration(args...) = DefaultEquilibration{DefaultFloat}(args...)


# ---------------
# residuals
# ---------------

mutable struct DefaultResiduals{T} <: AbstractResiduals{T}

    #the main KKT residuals
    rx::Vector{T}
    rz::Vector{T}
    rτ::T

    #partial residuals for infeasibility checks
    rx_inf::Vector{T}
    rz_inf::Vector{T}

    #various inner products.
    #NB: these are invariant w.r.t equilibration
    dot_qx::T
    dot_bz::T
    dot_sz::T
    dot_xPx::T

    #the product Px by itself. Required for infeasibilty checks
    Px::Vector{T}

    function DefaultResiduals{T}(n::Integer,
                                 m::Integer) where {T}

        rx = Vector{T}(undef,n)
        rz = Vector{T}(undef,m)
        rτ = T(1)

        rx_inf = Vector{T}(undef,n)
        rz_inf = Vector{T}(undef,m)

        Px = Vector{T}(undef,n)

        new(rx,rz,rτ,rx_inf,rz_inf,zero(T),zero(T),zero(T),zero(T),Px)
    end

end

DefaultResiduals(args...) = DefaultResiduals{DefaultFloat}(args...)


# ---------------
# problem data
# ---------------

mutable struct DefaultProblemData{T} <: AbstractProblemData{T}

    P::AbstractMatrix{T}
    q::Vector{T}
    A::AbstractMatrix{T}
    b::Vector{T}
    n::DefaultInt
    m::DefaultInt
    equilibration::DefaultEquilibration{T}

    normq::T
    normb::T

    presolver::Presolver{T}

    function DefaultProblemData{T}(
        P::AbstractMatrix{T},
        q::AbstractVector{T},
        A::AbstractMatrix{T},
        b::AbstractVector{T},
        cones::CompositeCone{T},
        presolver::Presolver{T},
    ) where {T}

        # dimension checks will have already been
        # performed during problem setup, so skip here

        #take an internal copy of all problem
        #data, since we are going to scale it
        P = triu(P)
        q = deepcopy(q)

        (A,b) = let  
            map = presolver.reduce_map;  
            if !isnothing(map)
                (
                    A[map.keep_logical,:],
                    b[map.keep_logical]
                )
            else
                (deepcopy(A),deepcopy(b))
            end
        end 

        #cap entries in b at INFINITY.  This is important 
        #for inf values that were not in a reduced cone
        @. b .= min(b,T(presolver.infbound))

        #this ensures m is the *reduced* size m
        (m,n) = size(A)

        equilibration = DefaultEquilibration{T}(n,cones)

        normq = norm(q, Inf)
        normb = norm(b, Inf)

        new(P,q,A,b,n,m,equilibration,normq,normb,presolver)

    end

end

DefaultProblemData(args...) = DefaultProblemData{DefaultFloat}(args...)


# ----------------------
# progress info
# ----------------------

mutable struct DefaultInfo{T} <: AbstractInfo{T}

    μ::T
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

    # previous iterate
    prev_cost_primal::T
    prev_cost_dual::T
    prev_res_primal::T
    prev_res_dual::T
    prev_gap_abs::T
    prev_gap_rel::T

    solve_time::Float64
    status::SolverStatus

    function DefaultInfo{T}() where {T}

        #here we set the first set of fields to zero (it doesn't matter),
        #but the previous iterates to Inf to avoid weird edge cases 
        prevvals = ntuple(x->floatmax(T), 6);
        new((ntuple(x->0, fieldcount(DefaultInfo)-6-1)...,prevvals...,UNSOLVED)...)
    end

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
obj_val | T | Objective value
solve_time | T | Solver run time
iterations | Int | Number of solver iterations
r_prim       | primal residual at termination
r_dual       | dual residual at termination

If the status field indicates that the problem is solved then (x,z,s) are the calculated solution, or a best guess if the solver has terminated early due to time or iterations limits.

If the status indicates either primal or dual infeasibility, then (x,z,s) provide instead an infeasibility certificate.
"""
mutable struct DefaultSolution{T} <: AbstractSolution{T}
    x::Vector{T}
    z::Vector{T}
    s::Vector{T}
    status::SolverStatus
    obj_val::T
    solve_time::T
    iterations::UInt32
    r_prim::T
    r_dual::T

end

function DefaultSolution{T}(m,n) where {T <: AbstractFloat}

    x = Vector{T}(undef,n)
    z = Vector{T}(undef,m)
    s = Vector{T}(undef,m)

    # seemingly reasonable defaults
    status  = UNSOLVED
    obj_val = T(NaN)
    solve_time = zero(T)
    iterations = 0
    r_prim     = T(NaN)
    r_dual     = T(NaN)

  return DefaultSolution{T}(x,z,s,status,obj_val,solve_time,iterations,r_prim,r_dual)
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

    data::Union{AbstractProblemData{T},Nothing}
    variables::Union{AbstractVariables{T},Nothing}
    cones::Union{CompositeCone{T},Nothing}
    residuals::Union{AbstractResiduals{T},Nothing}
    kktsystem::Union{AbstractKKTSystem{T},Nothing}
    info::Union{AbstractInfo{T},Nothing}
    step_lhs::Union{AbstractVariables{T},Nothing}
    step_rhs::Union{AbstractVariables{T},Nothing}
    prev_vars::Union{AbstractVariables{T},Nothing}
    solution::Union{AbstractSolution{T},Nothing}
    settings::Settings{T}
    timers::TimerOutput

end

#initializes all fields except settings to nothing
function Solver{T}(settings::Settings{T}) where {T}

    to = TimerOutput()
    #setup the main timer sections here and
    #zero them.   This ensures that the sections
    #exists if we try to clear them later
    @timeit to "setup!" begin (nothing) end
    @timeit to "solve!" begin (nothing) end
    reset_timer!(to["setup!"])
    reset_timer!(to["solve!"])

    Solver{T}(ntuple(x->nothing, fieldcount(Solver)-2)...,settings,to)
end

function Solver{T}() where {T}
    #default settings
    Solver{T}(Settings{T}())
end

#partial user defined settings
function Solver{T}(d::Dict) where {T}
    Solver{T}(Settings{T}(d))
end

Solver(args...; kwargs...) = Solver{DefaultFloat}(args...; kwargs...)
