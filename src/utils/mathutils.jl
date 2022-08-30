import LinearAlgebra: dot


function clip(
    s::Real,
    min_thresh::Real,
    max_thresh::Real,
    min_new::Real = min_thresh,
    max_new::Real = max_thresh)
	s = ifelse(s < min_thresh, min_new, ifelse(s > max_thresh, max_new, s))
    return s
end

@inline function logsafe(v::T) where {T<:Real}
    if v < 0
        return -typemax(T)
    else 
        return log(v)
    end
end


#computes dot(z + αdz,s + αds) without intermediate allocation

@inline function dot_shifted(
    z::AbstractVector{T}, 
    s::AbstractVector{T},
    dz::AbstractVector{T}, 
    ds::AbstractVector{T},
    α::T
    ) where {T<:Real}

    @assert(length(z) == length(s))
    @assert(length(z) == length(dz))
    @assert(length(s) == length(ds))
    
    out = zero(T)
    @inbounds for i in eachindex(z) 
        zi = z[i] + α*dz[i]
        si = s[i] + α*ds[i]
        out += zi*si
    end 

    return out

end


#2-norm of the product M*v
function scaled_norm(M::Diagonal{T},v::AbstractVector{T}) where{T}
    return scaled_norm(M.diag,v)
end

#2-norm of the product a.*b
function scaled_norm(m::AbstractVector{T},v::AbstractVector{T}) where{T}
    t = zero(T)
    for i in eachindex(v)
        p  = m[i]*v[i]
        t += p*p
    end
    return sqrt(t)
end



function kkt_col_norms!(
    P::AbstractMatrix{T},
    A::AbstractMatrix{T},
    norm_LHS::AbstractVector{T},
    norm_RHS::AbstractVector{T}
) where {T}

	col_norms_sym!(norm_LHS, P)   # P can be triu
	col_norms_no_reset!(norm_LHS, A)       #incrementally from P norms
	row_norms!(norm_RHS, A)                      #same as column norms of A'

	return nothing
end

function col_norms!(
    norms::AbstractVector{Tf},
	A::SparseMatrixCSC{Tf,Ti}
) where {Tf <: AbstractFloat, Ti <: Integer}

	fill!(norms, zero(Tf))
    col_norms_no_reset!(norms,A)
    return nothing
end

function col_norms_no_reset!(
    norms::AbstractVector{Tf},
	A::SparseMatrixCSC{Tf,Ti};
) where {Tf <: AbstractFloat, Ti <: Integer}

	@inbounds for i = eachindex(norms)
		for j = A.colptr[i]:(A.colptr[i + 1] - 1)
			tmp = abs(A.nzval[j])
			norms[i] = norms[i] > tmp ? norms[i] : tmp
		end
	end
	return nothing
end

#column norms of a matrix assumed to be symmetric.
#this works even if only tril or triu part is supplied
#don't worry about inspecting diagonal elements twice
#since we are taking inf norms here anyway


function col_norms_sym!(
    norms::AbstractVector{Tf},
	A::SparseMatrixCSC{Tf,Ti};
    reset::Bool = true
) where {Tf <: AbstractFloat, Ti <: Integer}

    fill!(norms, zero(Tf))
    col_norms_sym_no_reset!(norms,A)
    return nothing

end

function col_norms_sym_no_reset!(
    norms::AbstractVector{Tf},
	A::SparseMatrixCSC{Tf,Ti}
) where {Tf <: AbstractFloat, Ti <: Integer}

	@inbounds for i = eachindex(norms)
		for j = A.colptr[i]:(A.colptr[i + 1] - 1)
			tmp = abs(A.nzval[j])
            r   = A.rowval[j]
			norms[i] = norms[i] > tmp ? norms[i] : tmp
            norms[r] = norms[r] > tmp ? norms[r] : tmp
		end
	end
	return nothing
end

function row_norms!(
    norms::AbstractVector{Tf},
	A::SparseMatrixCSC{Tf, Ti}
) where{Tf <: AbstractFloat, Ti <: Integer}

    fill!(norms, zero(Tf))
    return row_norms_no_reset!(norms,A)
	return nothing
end

function row_norms_no_reset!(
    norms::AbstractVector{Tf},
	A::SparseMatrixCSC{Tf, Ti};
	reset::Bool = true
) where{Tf <: AbstractFloat, Ti <: Integer}

    @inbounds for i = 1:(A.colptr[end] - 1)
		idx = A.rowval[i]
		tmp = abs(A.nzval[i])
		norms[idx] = norms[idx] > tmp ? norms[idx] : tmp
	end
	return nothing
end


function scalarmul!(A::SparseMatrixCSC, c::Real)
	A.nzval .*= c
end


function lrscale!(L::AbstractVector{T}, M::SparseMatrixCSC{T}, R::AbstractVector{T}) where {T <: AbstractFloat}

	m, n = size(M)
	Mnzval  = M.nzval
	Mrowval = M.rowval
	Mcolptr = M.colptr
	(m == length(L) && n == length(R)) || throw(DimensionMismatch())

	@inbounds for i = 1:n
		for j = Mcolptr[i]:(Mcolptr[i + 1] - 1)
	 		Mnzval[j] *= L[Mrowval[j]] * R[i]
		end
	end
	return M
end

function lscale!(L::AbstractVector{T}, M::SparseMatrixCSC{T}) where {T <: AbstractFloat}

	#NB : Same as:  @views M.nzval .*= L[M.rowval]
	#but this way allocates no memory at all and
	#is marginally faster
	m, n = size(M)
	(m == length(L)) || throw(DimensionMismatch())

	@inbounds for i = 1:(M.colptr[end] - 1)
	 		M.nzval[i] *= L[M.rowval[i]]
	end
	return M
end

function rscale!(M::SparseMatrixCSC{T}, R::AbstractVector{T}) where {T <: AbstractFloat}

	m, n = size(M)
	(n == length(R)) || throw(DimensionMismatch())

	@inbounds for i = 1:n, j = M.colptr[i]:(M.colptr[i + 1] - 1)
		 	M.nzval[j] *= R[i]
	end
	return M
end

# dense version 
function lrscale!(L::AbstractVector{T},M::Matrix{T},R::AbstractVector{T}) where {T <: AbstractFloat}

    m, n = size(M)
    (n == length(R)) || throw(DimensionMismatch())
    (m == length(L)) || throw(DimensionMismatch())

    @inbounds for i = 1:m, j = 1:n
            M[i,j] *= L[i]*R[j]
    end
    return M
end 


#Julia SparseArrays dot function is very slow for Symmtric
#matrices.  See https://github.com/JuliaSparse/SparseArrays.jl/issues/83
function quad_form(
    x::AbstractArray{Tf},
    A::Symmetric{Tf,SparseMatrixCSC{Tf,Ti}},
    y::AbstractArray{Tf}
    ) where{Tf <: Real,Ti}

    if(A.uplo != 'U')
        error("Only implemented for upper triangular matrices")
    end
    M = A.data

    m, n = size(A)
    (length(x) == m && n == length(y)) || throw(DimensionMismatch())
    if iszero(m) || iszero(n)
        return zero(Tf)
    end

    Mc = M.colptr
    Mr = M.rowval
    Mv = M.nzval

    out = zero(Tf)

    @inbounds for j = 1:n    #col number
        tmp1 = zero(Tf)
        tmp2 = zero(Tf)
        for p = Mc[j]:(Mc[j+1]-1)
            i = Mr[p]  #row number
            if (i < j)  #triu terms only
                tmp1 += Mv[p]*x[i]
                tmp2 += Mv[p]*y[i]
            elseif i == j
                out += Mv[p]*x[i]*y[i]
            end
        end
        out += tmp1*y[j] + tmp2*x[j]
    end
    return out
end



# ---------------------------------
# functions for manipulating scaled vectors
# representing packed matrices in the upper
# triangle, read columnwise
# ---------------------------------
_triangle_svec_to_unscaled(v::T,idx::Int) where {T} = _triangle_svec_scale(v, idx, 1/sqrt(T(2)))
_triangle_unscaled_to_svec(v::T,idx::Int) where {T} = _triangle_svec_scale(v, idx,   sqrt(T(2)))
_triangle_svec_scale(v, index, scale) = _is_triangular_value(index) ? v : scale*v

#vectorized versions on full triangles
_triangle_svec_to_unscaled(v::AbstractVector) = _triangle_svec_to_unscaled.(v,eachindex(v))
_triangle_unscaled_to_svec(v::AbstractVector) = _triangle_unscaled_to_svec.(v,eachindex(v))

function _is_triangular_value(k::Int)
    #true if the int is a triangular number
    return isqrt(8*k + 1)^2 == 8*k + 1
end

#Just put elements from a vector of length
#n*(n+1)/2 into the upper triangle of an
#nxn matrix.   It does NOT perform any scaling
#of the vector entries.
function _pack_triu(v::Vector{T},A::Matrix{T}) where T
    n     = LinearAlgebra.checksquare(A)
    numel = (n*(n+1))>>1
    length(v) == numel || throw(DimensionMismatch())
    k = 1
    for col = 1:n, row = 1:col
        @inbounds v[k] = A[row,col]
        k += 1
    end
    return v
end

function _pack_triu(v::Vector{T},A::SparseMatrixCSC{T}) where T
    n     = 3
    k = 1
    for col = 1:n, row = 1:col
        @inbounds v[k] = A[row,col]
        k += 1
    end
    return v
end


#make a matrix view from a vectorized input
function _svec_to_mat!(M::AbstractMatrix{T}, x::AbstractVector{T}, K::PSDTriangleCone{T}) where {T}

    ISQRT2 = inv(sqrt(T(2)))

    idx = 1
    for col = 1:K.n, row = 1:col
        if row == col
            M[row,col] = x[idx]
            else
            M[row,col] = x[idx]*ISQRT2
            M[col,row] = x[idx]*ISQRT2
        end
        idx += 1
    end
end


function _mat_to_svec!(x::AbstractVector{T},M::AbstractMatrix{T},K::PSDTriangleCone{T}) where {T}

    ISQRT2 = 1/sqrt(T(2))

    idx = 1
    for row = 1:K.n, col = 1:row
        @inbounds x[idx] = row == col ? M[row,col] : (M[row,col]+M[col,row])*ISQRT2
        idx += 1
    end

    return nothing
end


#------------------------------
# special methods for solving 3x3 positive definite systems 
#------------------------------


# Unrolled 3x3 cholesky decomposition without pivoting 
# Returns `false` for a non-positive pivot and the 
# factorization is not completed
#
# NB: this is only marginally slower than the explicit
# 3x3 LDL decomposition, which would avoid sqrts.  

function cholesky_3x3_explicit_factor!(L,A)

    t = A[1,1]

    if t <= 0; return false; end

    L[1,1] = sqrt(A[1,1])
    L[2,1] = A[2,1]/L[1,1]

    t = A[2,2] - L[2,1]*L[2,1]

    if(t <= 0); return false; end

    L[2,2] = sqrt(t);
    L[3,1] = A[3,1] / L[1,1]
    L[3,2] = (A[3,2] - L[2,1]*L[3,1]) / L[2,2]

    t = A[3,3] - L[3,1]*L[3,1] - L[3,2]*L[3,2]

    if(t <= 0); return false; end
    L[3,3] = sqrt(t)

    return true

end

# Unrolled 3x3 forward/backward substition for a Cholesky factor

function cholesky_3x3_explicit_solve!(x,L,b)

  c1 = b[1]/L[1,1]
  c2 = (b[2]*L[1,1] - b[1]*L[2,1])/(L[1,1]*L[2,2])
  c3 = (b[3]*L[1,1]*L[2,2] - b[2]*L[1,1]*L[3,2] + b[1]*L[2,1]*L[3,2] - b[1]*L[2,2]*L[3,1])/(L[1,1]*L[2,2]*L[3,3])

 
 x[1] = (c1*L[2,2]*L[3,3] - c2*L[2,1]*L[3,3] + c3*L[2,1]*L[3,2] - c3*L[2,2]*L[3,1])/(L[1,1]*L[2,2]*L[3,3])
 x[2] = (c2*L[3,3] - c3*L[3,2])/(L[2,2]*L[3,3])
 x[3] = c3/L[3,3]
end

#------------------------------
# methods and types for indexing into the upper triangle of a square matrix
#------------------------------



struct TriuIndex <: AbstractVector{Int}
    n::Int
end

TriuIndex(A::AbstractMatrix) = TriuIndex(LinearAlgebra.checksquare(A))

function _get_triu_index(n,k)
    d = (isqrt(k*8 + 1) - 1)>>1
    r  = k - (d*(d+1))>>1
    return  r == 0 ? (d-1)*n+d : d*n+r
end

function Base.iterate(TI::TriuIndex, state = 1)
    return state > length(TI) ? nothing : (_get_triu_index(TI.n,state),state+1)
end

Base.eltype(::Type{TriuIndex}) = Int
Base.length(TI::TriuIndex) = ((TI.n+1)*TI.n)>>1
Base.firstindex(TI::TriuIndex) = 1
Base.lastindex(TI::TriuIndex) = length(TI)
Base.size(TI::TriuIndex) = (length(TI),)

function Base.getindex(TI::TriuIndex, i::Int)
    1 <= i <= length(TI) || throw(BoundsError(TI, i))
    return _get_triu_index(TI.n,i)
end

