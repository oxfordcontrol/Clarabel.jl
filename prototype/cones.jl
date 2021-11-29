
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

	#internal working variables
	w::Vector{T}

	function NonnegativeCone{T}(dim) where {T}

		dim >= 1 || throw(DomainError(dim, "dimension must be positive"))
		w = Vector{T}(undef,dim)
		new(dim,w)

	end
end

NonnegativeCone(args...) = NonnegativeCone{DefaultFloat}(args...)

# ----------------------------------------------------
# Second Order Cone
# ----------------------------------------------------

struct SecondOrderCone{T} <: AbstractCone{T}
	dim::DefaultInt
	function SecondOrderCone{T}(dim::Integer) where {T}
		dim >= 2 ? new(dim) : throw(DomainError(dim, "dimension must be >= 2"))
	end
end

SecondOrderCone(args...) = SecondOrderCone{DefaultFloat}(args...)


# -------------------------------------
# Enum and dict for user interface
# -------------------------------------

@enum SupportedCones begin
	ZeroConeT
	NonnegativeConeT
	SecondOrderT
end

const ConeDict = Dict(
	ZeroConeT =>  ZeroCone,
	NonnegativeConeT =>  NonnegativeCone,
	SecondOrderT     =>  SecondOrderCone
)

mutable struct ConeInfo

	# container for the type and size of each cone
    types::Vector{SupportedCones}
    dims::Vector{Int}

    function ConeInfo(types,dims)
        new(types,dims)
    end
end
