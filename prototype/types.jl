# -------------------------------------
# abstract type defs
# -------------------------------------
abstract type AbstractVariables{T <: AbstractFloat}   end
abstract type AbstractConeScalings{T <: AbstractFloat}   end
abstract type AbstractResiduals{T <: AbstractFloat}   end
abstract type AbstractProblemData{T <: AbstractFloat} end
abstract type AbstractKKTSolver{T <: AbstractFloat} end
abstract type AbstractInfo{T <: AbstractFloat} end
abstract type AbstractCone{T} end


# -------------------------------------
# vectors defined w.r.t. to conic constraints
# get this type with views into the subcomponents
# ---------------------------------------

mutable struct SplitVector{T} <: AbstractVariables{T}

    #contiguous array of source data
    vec::Vector{T}

    #array of data views of type Vector{T}
    views::Vector{VectorView{T}}

    function SplitVector{T}(
        cone_info::ConeInfo) where {T}

        vec   = Vector{T}(undef,cone_info.totaldim)
        views = Vector{VectorView{T}}(undef, length(cone_info.types))

        # loop over the sets and create views
        last = 0
        for i = eachindex(cone_info.dims)
            first  = last + 1
            last   = last + cone_info.dims[i]
            rng = first:last
            views[i] = view(vec, rng)
        end

        return new(vec, views)

    end

end

SplitVector(args...) = SplitVector{DefaultFloat}(args...)


# -------------------------------------
# default solver subcomponent implementations
# -------------------------------------

# ---------------
# variables
# ---------------

mutable struct DefaultVariables{T} <: AbstractVariables{T}

    x::Vector{T}
    s::SplitVector{T}
    z::SplitVector{T}
    τ::T
    κ::T

    function DefaultVariables{T}(
        n::Integer,
        cone_info::ConeInfo) where {T}

        x = Vector{T}(undef,n)
        s = SplitVector{T}(cone_info)
        z = SplitVector{T}(cone_info)
        τ = T(1)
        κ = T(1)

        new(x,s,z,τ,κ)
    end

end

DefaultVariables(args...) = DefaultVariables{DefaultFloat}(args...)


# ---------------
# scalings
# ---------------

mutable struct DefaultScalings{T} <: AbstractConeScalings{T}

    # specification from the problem statement
    cone_info::ConeInfo

    # vector of objects implementing the scalings
    cones::ConeSet{T}

    # scaled variable λ = Wz = W^{-1}s
    λ::SplitVector{T}

    #composite cone order.  NB: Not the
    #same as dimension for zero or SO cones
    total_order::DefaultInt

end

DefaultScalings(args...) = DefaultScalings{DefaultFloat}(args...)


# ---------------
# residuals
# ---------------

mutable struct DefaultResiduals{T} <: AbstractResiduals{T}

    rx::Vector{T}
    rz::Vector{T}
    rτ::T

    norm_Ax::T
    norm_Atz::T

    norm_rz::T
    norm_rx::T

    #various inner products
    dot_cx::T
    dot_bz::T
    dot_sz::T
    dot_xPx::T

    function DefaultResiduals{T}(n::Integer,
                                 m::Integer) where {T}

        rx = Vector{T}(undef,n)
        rz = Vector{T}(undef,m)
        rτ = T(1)

        new(rx,rz,rτ)
    end

end

DefaultResiduals(args...) = DefaultResiduals{DefaultFloat}(args...)


# ---------------
# problem data
# ---------------

mutable struct DefaultProblemData{T} <: AbstractProblemData{T}

    P::AbstractMatrix{T}
    c::Vector{T}
    A::AbstractMatrix{T}
    b::Vector{T}
    n::DefaultInt
    m::DefaultInt
    cone_info::ConeInfo

    #some static info about the problem data
    norm_c::T
    norm_b::T

    function DefaultProblemData{T}(P,c,A,b,cone_info) where {T}

        n         = length(c)
        m         = length(b)

        m == size(A)[1] || throw(ErrorException("A and b incompatible dimensions."))
        n == size(A)[2] || throw(ErrorException("A and c incompatible dimensions."))
        m == sum(cone_info.dims) || throw(ErrorException("Incompatible cone dimensions."))

        new(P,c,A,b,n,m,cone_info,norm(c),norm(b))

    end

end

DefaultProblemData(args...) = DefaultProblemData{DefaultFloat}(args...)

# ---------------
# solver status
# ---------------

@enum SolverStatus begin
    UNSOLVED           = 0
    SOLVED
    PRIMAL_INFEASIBLE
    DUAL_INFEASIBLE
    MAX_ITERATIONS
end

const SolverStatusDict = Dict(
    UNSOLVED    =>  "unsolved",
    SOLVED      =>  "solved",
    PRIMAL_INFEASIBLE =>  "primal infeasible",
    DUAL_INFEASIBLE =>  "dual infeasible",
    MAX_ITERATIONS  =>  "iteration limit"
)

mutable struct DefaultInfo{T} <: AbstractInfo{T}

    cost_primal::T
    cost_dual::T
    res_primal::T
    res_dual::T
    gap::T
    step_length::T
    sigma::T
    ktratio::T
    iterations::DefaultInt
    solve_time::T
    status::SolverStatus

    function DefaultInfo{T}() where {T}
        new( (ntuple(x->0, fieldcount(DefaultInfo)-1)...,UNSOLVED)...)
    end

end

DefaultInfo(args...) = DefaultInfo{DefaultFloat}(args...)

# -------------------------------------
# top level solver type
# -------------------------------------

mutable struct Solver{T <: AbstractFloat}

    data::Union{AbstractProblemData{T},Nothing}
    variables::Union{AbstractVariables{T},Nothing}
    scalings::Union{AbstractConeScalings{T},Nothing}
    residuals::Union{AbstractResiduals{T},Nothing}
    kktsolver::Union{AbstractKKTSolver{T},Nothing}
    info::Union{AbstractInfo{T},Nothing}
    settings::Union{Settings{T},Nothing}
    step_lhs::Union{AbstractVariables{T},Nothing}
    step_rhs::Union{AbstractVariables{T},Nothing}

end

#initializes all fields to nothing
Solver{DefaultFloat}() = Solver{DefaultFloat}(ntuple(x->nothing, fieldcount(Solver))...)

Solver(args...) = Solver{DefaultFloat}(args...)
