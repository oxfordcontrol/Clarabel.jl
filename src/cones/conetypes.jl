# -------------------------------------
# abstract type defs
# -------------------------------------
abstract type AbstractCone{T} end

function Base.deepcopy(m::Type{<: AbstractCone{T}}) where {T}
    typeof(m)(deepcopy(m.dim))
end

# -------------------------------------
# Zero Cone
# -------------------------------------

struct ZeroCone{T} <: AbstractCone{T}

    dim::DefaultInt

    function ZeroCone{T}(dim::Integer) where {T}
        dim >= 1 || throw(DomainError(dim, "dimension must be positive"))
        new(dim)
    end

end

ZeroCone(args...) = ZeroCone{DefaultFloat}(args...)


# ------------------------------------
# Nonnegative Cone
# -------------------------------------

struct NonnegativeCone{T} <: AbstractCone{T}

    dim::DefaultInt

    #internal working variables for W and λ
    w::Vector{T}
    λ::Vector{T}

    function NonnegativeCone{T}(dim) where {T}

        dim >= 1 || throw(DomainError(dim, "dimension must be positive"))
        w = Vector{T}(undef,dim)
        λ = Vector{T}(undef,dim)
        return new(dim,w,λ)

    end

end

NonnegativeCone(args...) = NonnegativeCone{DefaultFloat}(args...)

# ----------------------------------------------------
# Second Order Cone
# ----------------------------------------------------

mutable struct SecondOrderCone{T} <: AbstractCone{T}

    dim::DefaultInt

    #internal working variables for W and its products
    w::Vector{T}

    #scaled version of (s,z)
    λ::Vector{T}

    #vectors for rank 2 update representation of W^2
    u::Vector{T}
    v::Vector{T}

    #additional scalar terms for rank-2 rep
    d::T
    η::T

    function SecondOrderCone{T}(dim::Integer) where {T}
        dim >= 2 ? new(dim) : throw(DomainError(dim, "dimension must be >= 2"))
        w = Vector{T}(undef,dim)
        λ = Vector{T}(undef,dim)
        u = Vector{T}(undef,dim)
        v = Vector{T}(undef,dim)
        d = one(T)
        η = zero(T)
        return new(dim,w,λ,u,v,d,η)
    end

end

SecondOrderCone(args...) = SecondOrderCone{DefaultFloat}(args...)

# ------------------------------------
# Positive Semidefinite Cone
# ------------------------------------

mutable struct PSDConeWork #PJG: where {T}...

    cholS
    cholZ
    SVD
    U
    V
    λ
    R
    Rinv
    L1

    L2
    PSDConeWork() = new(ntuple(x->nothing, fieldcount(PSDConeWork)))
end

#PJG: PSDConeWork(args...) = PSDConeWork{DefaultFloat}(args...)

struct PSDCone{T} <: AbstractCone{T}

    dim::DefaultInt  #this is the total number of elements in the matrix
      n::DefaultInt  #this is the matrix dimension, i.e. n^2 = dim

    #internal working variables for W and λ
    λ::Vector{T}     #NB: not a matrix b/c scaled (S,Z) should be diagonal
    R::Matrix{T}     #PJG: R is tril factor of W = R*R'
    work::PSDConeWork   #PJG: kludgey AF for now

    #PJG: need some further structure here to maintain
    #working memory for all of the steps in computing R

    function PSDCone{T}(dim) where {T}

        dim >= 1   || throw(DomainError(dim, "dimension must be positive"))
        n = isqrt(dim)
        n*n == dim || throw(DomainError(dim, "dimension must be a square"))

        #PJG: R should really be tril part only.  Square
        #for initial debugging purposes
        λ = Vector{T}(undef,n)
        R = Matrix{T}(undef,n,n)
        work = PSDConeWork()

        return new(dim,n,λ,R,work)

    end

end

PSDCone(args...) = PSDCone{DefaultFloat}(args...)

# -------------------------------------
# collection of cones for composite
# operations on a compound set
# -------------------------------------

const ConeSet{T} = Vector{AbstractCone{T}}


# -------------------------------------
# Enum and dict for user interface
# -------------------------------------
"""
    SupportedCones
An Enum of supported cone type for passing to [`setup!`](@ref). The currently
supported types are:

* `ZeroConeT`       : The zero cone.  Used to define equalities.
* `NonnegativeConeT`: The nonnegative orthant.
* `SecondOrderConeT`: The second order / Lorentz / ice-cream cone.
# `PSDConeT`        : The positive semidefinite cone.
"""
@enum SupportedCones begin
    ZeroConeT
    NonnegativeConeT
    SecondOrderConeT
    PSDConeT
end

"""
    ConeDict
A Dict that maps the user-facing SupportedCones enum values to
the types used internally in the solver.   See [SupportedCones](@ref)
"""
const ConeDict = Dict(
           ZeroConeT => ZeroCone,
    NonnegativeConeT => NonnegativeCone,
    SecondOrderConeT => SecondOrderCone,
            PSDConeT => PSDCone,
)

mutable struct ConeInfo

    # container for the type and size of each cone
    types::Vector{SupportedCones}
    dims::Vector{Int}

    #Count of each cone type.
    type_counts::Dict{SupportedCones,Int}

    #total dimension
    totaldim::DefaultInt

    #a vector showing the overall index of the
    #first element in each cone.  For convenience
    headidx::Vector{Int}

    function ConeInfo(types,dims)

        #count the number of each cone type
        type_counts = Dict{SupportedCones,Int}()
        for coneT in instances(SupportedCones)
            type_counts[coneT] = count(==(coneT), types)
        end

        headidx = Vector{Int}(undef,length(dims))
        if(length(dims) > 0)
            #index of first element in each cone
            headidx[1] = 1
            for i = 2:length(dims)
                headidx[i] = headidx[i-1] + dims[i-1]
            end

            #total dimension of all cones together
            totaldim = headidx[end] + dims[end] - 1
        else
            totaldim = 0
        end

        return new(types,dims,type_counts,totaldim,headidx)
    end
end
