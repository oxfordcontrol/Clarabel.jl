# -------------------------------------
# abstract type defs
# -------------------------------------
abstract type AbstractVariables{T <: AbstractFloat}   end
abstract type AbstractConeScalings{T <: AbstractFloat}   end
abstract type AbstractResiduals{T <: AbstractFloat}   end
abstract type AbstractProblemData{T <: AbstractFloat} end
abstract type AbstractKKTSolver{T <: AbstractFloat} end
abstract type AbstractStatus{T <: AbstractFloat} end
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
        n::Integer,
        cone_info::ConeInfo) where {T}

        vec   = Vector{T}(undef,n)
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
    #PJG: Not convinced that this belongs here
    λ::SplitVector{T}

    #PJG:I don't think lambda belongs here because
    #there is no lambda required for the step directions

    function DefaultVariables{T}(
        n::Integer,
        m::Integer,
        cone_info::ConeInfo) where {T}

        x = Vector{T}(undef,n)
        s = SplitVector{T}(m,cone_info)
        z = SplitVector{T}(m,cone_info)
        τ = T(1)
        κ = T(1)
        λ = SplitVector{T}(m,cone_info)

        new(x,s,z,τ,κ,λ)
    end

end

DefaultVariables(args...) = DefaultVariables{DefaultFloat}(args...)


# ---------------
# scalings
# ---------------

mutable struct DefaultConeScalings{T} <: AbstractConeScalings{T}

    # specification from the problem statement
    cone_info::ConeInfo

    # vector of objects containing the scalings
    cones::Vector{AbstractCone{T}}

    #composite cone sizes
    #not convinced that these belong here.  Maybe in problem data
    #where it can also be checked for dimensional compatibility
    total_dim::DefaultInt
    total_order::DefaultInt

end

DefaultConeScalings(args...) = DefaultConeScalings{DefaultFloat}(args...)




# ---------------
# residuals
# ---------------

#PJG: NB -- struct is identical to Variables structure

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

    c::Vector{T}
    A::AbstractMatrix{T}
    b::Vector{T}
    n::DefaultInt
    m::DefaultInt
    cone_info::ConeInfo

    #some static info about the problem data
    norm_c::T
    norm_b::T

    function DefaultProblemData{T}(c,A,b,cone_info) where {T}
        n         = length(c)
        m         = length(b)
        #PJG: dimension sanity checks here
        new(c,A,b,n,m,cone_info,norm(c),norm(b))
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

mutable struct DefaultStatus{T} <: AbstractStatus{T}

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

    function DefaultStatus{T}() where {T}
        #new(ntuple(x->0, fieldcount(DefaultStatus)-1),UNSOLVED...)
        new( (ntuple(x->0, fieldcount(DefaultStatus)-1)...,UNSOLVED)...)
    end

end

DefaultStatus(args...) = DefaultStatus{DefaultFloat}(args...)

# -------------------------------------
# top level solver type
# -------------------------------------

mutable struct Solver{T <: AbstractFloat}

    data::Union{AbstractProblemData{T},Nothing}
    variables::Union{AbstractVariables{T},Nothing}
    scalings::Union{AbstractConeScalings{T},Nothing}
    residuals::Union{AbstractResiduals{T},Nothing}
    kktsolver::Union{AbstractKKTSolver{T},Nothing}
    status::Union{AbstractStatus{T},Nothing}
    settings::Union{Settings{T},Nothing}
    step_lhs::Union{AbstractVariables{T},Nothing}
    step_rhs::Union{AbstractVariables{T},Nothing}

end

#initializes all fields to nothing
Solver{DefaultFloat}() = Solver{DefaultFloat}(ntuple(x->nothing, fieldcount(Solver))...)

Solver(args...) = Solver{DefaultFloat}(args...)
