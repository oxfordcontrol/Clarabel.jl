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
        n::Integer, cones::ConeSet) where {T}

        x = Vector{T}(undef,n)
        s = ConicVector{T}(cones)
        z = ConicVector{T}(cones)
        τ = T(1)
        κ = T(1)

        new(x,s,z,τ,κ)
    end

end

DefaultVariables(args...) = DefaultVariables{DefaultFloat}(args...)


# ---------------
# equilibration data
# ---------------

struct DefaultEquilibration{T} <: AbstractEquilibration{T}

    #scaling matrices for problem data equilibration
    #fields d,e,dinv,einv are vectors of scaling values
    #The other fields are diagonal views for convenience
    d::Vector{T}
    dinv::Vector{T}
    D::Diagonal{T}
    Dinv::Diagonal{T}

    e::ConicVector{T}
    einv::ConicVector{T}
    E::Diagonal{T}
    Einv::Diagonal{T}

    #overall scaling for objective function
    c::Base.RefValue{T}

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

    #the product Px by itself required infeasibilty checks
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

    # we will require products P*x, but will only store triu(P).
    # Use this convenience object for symmetric products etc
    Psym::AbstractMatrix{T}

    function DefaultProblemData{T}(
        P::AbstractMatrix{T},
        q::AbstractVector{T},
        A::AbstractMatrix{T},
        b::AbstractVector{T},
    ) where {T}

        n = length(q)
        m = length(b)

        m == size(A)[1] || throw(DimensionMismatch("A and b incompatible dimensions."))
        n == size(A)[2] || throw(DimensionMismatch("A and q incompatible dimensions."))
        n == size(P)[1] || throw(DimensionMismatch("P and q incompatible dimensions."))
        size(P)[1] == size(P)[2] || throw(DimensionMismatch("P not square."))

        #take an internal copy of all problem
        #data, since we are going to scale it
        P = triu(P)
        Psym = Symmetric(P)
        A = deepcopy(A)
        q = deepcopy(q)
        b = deepcopy(b)

        new(P,q,A,b,n,m,Psym)

    end

end

DefaultProblemData(args...) = DefaultProblemData{DefaultFloat}(args...)


# ---------------
# data equilibration
# ---------------

function DefaultEquilibration{T}(
    nvars::Int,
    cones::ConeSet{T},
    settings::Settings
) where {T}

    #Left/Right diagonal scaling for problem data
    d    = Vector{T}(undef,nvars)
    dinv = Vector{T}(undef,nvars)
    D    = Diagonal(d)
    Dinv = Diagonal(dinv)

    e    = ConicVector{T}(cones)
    einv = ConicVector{T}(cones)
    E    = Diagonal(e)
    Einv = Diagonal(einv)

    c    = Ref(T(1.))

    return DefaultEquilibration(
            d,dinv,D,Dinv,e,einv,E,Einv,c
           )
end


# ---------------
# solver status
# ---------------
"""
    SolverStatus
An Enum of of possible conditions set by [`solve!`](@ref).

If no call has been made to [`solve!`](@ref), then the `SolverStatus`
is:
* `UNSOLVED`: The algorithm has not started.

Otherwise:
* `SOLVED`              : Solver as terminated with a solution.
* `PRIMAL_INFEASIBLE`   : Problem is primal infeasible.  Solution returned is a certificate of primal infeasibility.
* `DUAL_INFEASIBLE`     : Problem is dual infeasible.  Solution returned is a certificate of dual infeasibility.
* `MAX_ITERATIONS`      : Iteration limit reached before solution or infeasibility certificate found.
* `MAX_TIME`            : Time limit reached before solution or infeasibility certificate found.
"""
@enum SolverStatus begin
    UNSOLVED           = 0
    SOLVED
    PRIMAL_INFEASIBLE
    DUAL_INFEASIBLE
    MAX_ITERATIONS
    MAX_TIME
end

const SolverStatusDict = Dict(
    UNSOLVED            =>  "unsolved",
    SOLVED              =>  "solved",
    PRIMAL_INFEASIBLE   =>  "primal infeasible",
    DUAL_INFEASIBLE     =>  "dual infeasible",
    MAX_ITERATIONS      =>  "iteration limit",
    MAX_TIME            =>  "time limit"
)

mutable struct DefaultInfo{T} <: AbstractInfo{T}

    μ::T
    sigma::T
    step_length::T
    iterations::DefaultInt
    cost_primal::T
    cost_dual::T
    res_primal::T
    res_dual::T
    res_primal_inf::T
    res_dual_inf::T
    gap_abs::T
    gap_rel::T
    ktratio::T
    solve_time::T
    timer::TimerOutput
    status::SolverStatus

    function DefaultInfo{T}() where {T}

        to = TimerOutput()
        #setup the main timer sections here and
        #zero them.   This ensures that the sections
        #exists if we try to clear them later
        @timeit to "setup!" begin (nothing) end
        @timeit to "solve!" begin (nothing) end
        reset_timer!(to["setup!"])
        reset_timer!(to["solve!"])

        new( (ntuple(x->0, fieldcount(DefaultInfo)-2)...,to,UNSOLVED)...)
    end

end

DefaultInfo(args...) = DefaultInfo{DefaultFloat}(args...)

# -------------------------------------
# top level solver type
# -------------------------------------

"""
	Solver{T <: AbstractFloat}()
Initializes an empty Clarabel solver that can be filled with problem data using:

    setup!(solver, P, q, A, b, cone_types, cone_dims, [settings]).

"""
mutable struct Solver{T <: AbstractFloat}

    data::Union{AbstractProblemData{T},Nothing}
    variables::Union{AbstractVariables{T},Nothing}
    equilibration::Union{AbstractEquilibration{T},Nothing}
    cones::Union{ConeSet{T},Nothing}
    residuals::Union{AbstractResiduals{T},Nothing}
    kktsystem::Union{AbstractKKTSystem{T},Nothing}
    info::Union{AbstractInfo{T},Nothing}
    step_lhs::Union{AbstractVariables{T},Nothing}
    step_rhs::Union{AbstractVariables{T},Nothing}
    settings::Settings{T}

end

#initializes all fields except settings to nothing
function Solver{T}(settings::Settings{T}) where {T}
    Solver{T}(ntuple(x->nothing, fieldcount(Solver)-1)...,settings)
end

function Solver{T}() where {T}
    #default settings
    Solver{T}(Settings{T}())
end

#partial user defined settings
function Solver(d::Dict) where {T}
    Solver{T}(Settings(d))
end

Solver(args...; kwargs...) = Solver{DefaultFloat}(args...; kwargs...)
