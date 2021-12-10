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


## ------------------------------------
# Nonnegative Cone
# -------------------------------------

struct NonnegativeCone{T} <: AbstractCone{T}

    dim::DefaultInt

    #internal working variables for W
    w::Vector{T}

    function NonnegativeCone{T}(dim) where {T}

        dim >= 1 || throw(DomainError(dim, "dimension must be positive"))
        w = Vector{T}(undef,dim)
        return new(dim,w)

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

    #vectors for rank 2 update representation of W^2
    u::Vector{T}
    v::Vector{T}

    #additional scalar terms for rank-2 rep
    d::T
    η::T

    function SecondOrderCone{T}(dim::Integer) where {T}
        dim >= 2 ? new(dim) : throw(DomainError(dim, "dimension must be >= 2"))
        w = Vector{T}(undef,dim)
        u = Vector{T}(undef,dim)
        v = Vector{T}(undef,dim)
        d = 1.
        η = 0.
        return new(dim,w,u,v,d,η)
    end

end

SecondOrderCone(args...) = SecondOrderCone{DefaultFloat}(args...)

# -------------------------------------
# collection of cones for composite
# operations on a compound set
# -------------------------------------

const ConeSet{T} = Vector{AbstractCone{T}}


# -------------------------------------
# Enum and dict for user interface
# -------------------------------------

@enum SupportedCones begin
    ZeroConeT
    NonnegativeConeT
    SecondOrderConeT
end

const ConeDict = Dict(
           ZeroConeT => ZeroCone,
    NonnegativeConeT => NonnegativeCone,
    SecondOrderConeT => SecondOrderCone
)

mutable struct ConeInfo

    # container for the type and size of each cone
    types::Vector{SupportedCones}
    dims::Vector{Int}

    #Count of each cone type.
    k_zerocone::DefaultInt
    k_nncone::DefaultInt
    k_socone::DefaultInt

    #total dimension
    totaldim::DefaultInt

    function ConeInfo(types,dims)

        #count the number of each cone type
        k_zerocone = count(==(ZeroConeT),       types)
        k_nncone   = count(==(NonnegativeConeT),types)
        k_socone   = count(==(SecondOrderConeT),types)
        totaldim   = sum(dims)

        return new(types,dims,k_zerocone,k_nncone,k_socone,totaldim)
    end
end
