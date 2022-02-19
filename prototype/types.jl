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

mutable struct SplitVector{T}

    #contiguous array of source data
    vec::Vector{T}

    #array of data views of type Vector{T}
    views::Vector{VectorView{T}}

    function SplitVector{T}(
        cone_info::ConeInfo) where {T}

        vec   = Vector{T}(undef,cone_info.totaldim)
        #NB: failure to initialize here gives an error
        #because λ is updated using gemv! style update.
        #This fails if undefs include NaN entries
        vec  .= 0

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

struct DefaultScalings{T} <: AbstractConeScalings{T}

    # specification from the problem statement
    cone_info::ConeInfo

    # vector of objects implementing the scalings
    cones::ConeSet{T}

    # scaled variable λ = Wz = W^{-1}s
    λ::SplitVector{T}

    #composite cone degree.  NB: Not the
    #same as dimension for zero or SO cones
    total_degree::DefaultInt

    #scaling matrices for problem data equilibration
    #fields d,e,dinv,einv are vectors of scaling values
    #The other fields are diagonal views for convenience
    d::Vector{T}
    dinv::Vector{T}
    D::Diagonal{T}
    Dinv::Diagonal{T}

    e::SplitVector{T}
    einv::SplitVector{T}
    E::Diagonal{T}
    Einv::Diagonal{T}

    #overall scaling for objective function
    c::Ref{T}

end

DefaultScalings(args...) = DefaultScalings{DefaultFloat}(args...)


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

    function DefaultResiduals{T}(n::Integer,
                                 m::Integer) where {T}

        rx = Vector{T}(undef,n)
        rz = Vector{T}(undef,m)
        rτ = T(1)

        rx_inf = Vector{T}(undef,n)
        rz_inf = Vector{T}(undef,m)

        new(rx,rz,rτ,rx_inf,rz_inf,0.,0.,0.,0.)
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
    cone_info::ConeInfo

    # we will require products P*x, but will
    # only store triu(P).   Use this convenience
    # object for now
    Psym::AbstractMatrix{T}


    function DefaultProblemData{T}(P,q,A,b,cone_info) where {T}

        n         = length(q)
        m         = length(b)

        m == size(A)[1] || throw(ErrorException("A and b incompatible dimensions."))
        n == size(A)[2] || throw(ErrorException("A and c incompatible dimensions."))
        m == sum(cone_info.dims) || throw(ErrorException("Incompatible cone dimensions."))

        #take an internal copy of all problem
        #data, since we are going to scale it
        P = triu(P)
        Psym = Symmetric(P)
        A = deepcopy(A)
        q = deepcopy(q)
        b = deepcopy(b)

        new(P,q,A,b,n,m,cone_info,Psym)

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
    res_primal_inf::T
    res_dual_inf::T
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
