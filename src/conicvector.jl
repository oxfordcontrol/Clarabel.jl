# -------------------------------------
# vectors defined w.r.t. to conic constraints
# get this type with views into the subcomponents
# ---------------------------------------

const VectorView{T} = SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}

struct ConicVector{T<:AbstractFloat} <: AbstractVector{T}

    #contiguous array of source data
    vec::Vector{T}

    #array of data views of type Vector{T}
    views::Vector{VectorView{T}}

    function ConicVector{T}(S::CompositeCone{T}) where {T}

        #undef initialization would possibly result
        #in Infs or NaNs, causing failure in gemv!
        #style vector updates
        vec   = zeros(T,S.numel)
        views = Vector{VectorView{T}}(undef, length(S))

        # loop over the sets and create views
        last = 0
        for i = eachindex(S)
            first  = last + 1
            last   = last + numel(S[i])
            views[i] = view(vec, first:last)
        end

        return new(vec, views)

    end

end

ConicVector(args...) = ConicVector{DefaultFloat}(args...)




@inline function Base.getindex(s::ConicVector{T},i) where{T}
    @boundscheck checkbounds(s.vec,i)
    @inbounds s.vec[i]
end
@inline function Base.setindex!(s::ConicVector{T},v,i) where{T}
    @boundscheck checkbounds(s.vec,i)
    @inbounds s.vec[i] = T(v)
end

Base.adjoint(s::ConicVector{T}) where{T} = adjoint(s.vec)
Base.iterate(s::ConicVector{T}) where{T} = iterate(s.vec)
Base.eltype(s::ConicVector{T}) where{T} = eltype(s.vec)
Base.elsize(s::ConicVector{T}) where{T} = Base.elsize(s.vec)
Base.size(s::ConicVector{T}) where{T} = size(s.vec)
Base.length(s::ConicVector{T}) where{T} = length(s.vec)
Base.IndexStyle(s::ConicVector{T}) where{T} = IndexStyle(s.vec)

#For maximum speed, it seems we need to explicitly define
#a bunch of functions that use the vec field directly, which
#will force calls to the BLAS specialized methods when possible
#Alternatively, we could subtype DenseArray{T}, but that
#seems less general and still fails to capture high
#performance sparse matrix vector multiply


#need this if we want to make calls directly to the BLAS functions
Base.unsafe_convert(::Type{Ptr{T}}, s::ConicVector{T}) where {T} =
           Base.unsafe_convert(Ptr{T}, s.vec)

#dot
LinearAlgebra.dot(x::AbstractVector{T},y::ConicVector{T}) where {T} = dot(x,y.vec)
LinearAlgebra.dot(x::ConicVector{T},y::AbstractVector{T}) where {T} = dot(x.vec,y)
LinearAlgebra.dot(x::ConicVector{T},y::ConicVector{T}) where {T} = dot(x.vec,y.vec)

#mul!
LinearAlgebra.mul!(
    C::ConicVector,
    A::AbstractVecOrMat,
    B::AbstractVector, α::Number, β::Number) = mul!(C.vec, A, B, α, β)

LinearAlgebra.mul!(
    C::AbstractVector,
    A::AbstractVecOrMat,
    B::ConicVector, α::Number, β::Number) = mul!(C, A, B.vec, α, β)

LinearAlgebra.mul!(
    C::ConicVector,
    A::Adjoint{<:Any, <:AbstractVecOrMat},
    B::AbstractVector, α::Number, β::Number) = mul!(C.vec, A, B, α, β)

LinearAlgebra.mul!(
    C::AbstractVector,
    A::Adjoint{<:Any, <:AbstractVecOrMat},
    B::ConicVector, α::Number, β::Number) = mul!(C, A, B.vec, α, β)

LinearAlgebra.norm(x::ConicVector{T}) where {T} = norm(x.vec)
LinearAlgebra.norm(x::ConicVector{T},p::Real) where {T} = norm(x.vec,p)