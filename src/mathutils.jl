import LinearAlgebra: dot


function clip(
    s::Real,
    min_thresh::Real,
    max_thresh::Real,
    min_new::Real = min_thresh,
    max_new::Real = max_thresh)
	s = ifelse(s < min_thresh, min_new, ifelse(s > max_thresh, max_new, s))
end


function inv_sqrt!(v::AbstractVector{T}) where{T <: Real}
	@fastmath v .= one(T) ./ sqrt.(v)

    return nothing
end


#2-norm of the product M*v
function scaled_norm(M::Diagonal{T},v::AbstractVector{T}) where{T}
    return scaled_norm(M.diag,v)
end

#2-norm of the product a.*b
function scaled_norm(m::AbstractVector{T},v::AbstractVector{T}) where{T}
    t = 0.
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

	col_norms_sym!(norm_LHS, P, reset = true)   #start from zero.  P can be triu
	col_norms!(norm_LHS, A, reset = false)       #incrementally from P norms
	row_norms!(norm_RHS, A)                      #same as column norms of A'

	return nothing
end


function col_norms!(
    v::AbstractVector{Tf},
	A::SparseMatrixCSC{Tf,Ti};
    reset::Bool = true
) where {Tf <: AbstractFloat, Ti <: Integer}

	if reset
		fill!(v, 0)
	end

	@inbounds for i = eachindex(v)
		for j = A.colptr[i]:(A.colptr[i + 1] - 1)
			tmp = abs(A.nzval[j])
			v[i] = v[i] > tmp ? v[i] : tmp
		end
	end
	return v
end

#column norms of a matrix assumed to be symmetric.
#this works even if only tril or triu part is supplied
#don't worry about inspecting diagonal elements twice
#since we are taking inf norms here anyway

function col_norms_sym!(
    v::AbstractVector{Tf},
	A::SparseMatrixCSC{Tf,Ti};
    reset::Bool = true
) where {Tf <: AbstractFloat, Ti <: Integer}

	if reset
		fill!(v, 0)
	end

	@inbounds for i = eachindex(v)
		for j = A.colptr[i]:(A.colptr[i + 1] - 1)
			tmp = abs(A.nzval[j])
            r   = A.rowval[j]
			v[i] = v[i] > tmp ? v[i] : tmp
            v[r] = v[r] > tmp ? v[r] : tmp
		end
	end
	return v
end





function row_norms!(
    v::AbstractVector{Tf},
	A::SparseMatrixCSC{Tf, Ti};
	reset::Bool = true
) where{Tf <: AbstractFloat, Ti <: Integer}

	if reset
		fill!(v,0.)
	end

	@inbounds for i = 1:(A.colptr[end] - 1)
		idx = A.rowval[i]
		tmp = abs(A.nzval[i])
		v[idx] = v[idx] > tmp ? v[idx] : tmp
	end
	return v
end


function scalarmul!(A::SparseMatrixCSC, c::Real)
	A.nzval .*= c
end


function lrmul!(L::Diagonal{T}, M::SparseMatrixCSC{T}, R::Diagonal{T}) where {T <: AbstractFloat}

	m, n = size(M)
	Mnzval  = M.nzval
	Mrowval = M.rowval
	Mcolptr = M.colptr
	Rd      = R.diag
	Ld      = L.diag
	(m == length(Ld) && n == length(Rd)) || throw(DimensionMismatch())

	@inbounds for i = 1:n
		for j = Mcolptr[i]:(Mcolptr[i + 1] - 1)
	 		Mnzval[j] *= Ld[Mrowval[j]] * Rd[i]
		end
	end
	return M
end

function lmul!(L::Diagonal{T}, M::SparseMatrixCSC{T}) where {T <: AbstractFloat}

	#NB : Same as:  @views M.nzval .*= D.diag[M.rowval]
	#but this way allocates no memory at all and
	#is marginally faster
	m, n = size(M)
	(m == length(L.diag)) || throw(DimensionMismatch())

	@inbounds for i = 1:(M.colptr[end] - 1)
	 		M.nzval[i] *= L.diag[M.rowval[i]]
	end
	return M
end

lmul!(L::IdentityMatrix, M::AbstractMatrix) = L.λ ? M : M .= zero(eltype(M))

function lmul!(L::Diagonal{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
	(length(L.diag) == length(x)) || throw(DimensionMismatch())
	@. x = x * L.diag
	return nothing
end

lmul!(L::IdentityMatrix, x::AbstractVector{T}) where {T <: AbstractFloat} = L.λ ? x : x .= zero(eltype(x))



function rmul!(M::SparseMatrixCSC{T}, R::Diagonal{T}) where {T <: AbstractFloat}

	m, n = size(M)
	(n == length(R.diag)) || throw(DimensionMismatch())

	@inbounds for i = 1:n, j = M.colptr[i]:(M.colptr[i + 1] - 1)
		 	M.nzval[j] *= R.diag[i]
	end
	return M
end

rmul!(M::AbstractMatrix, R::IdentityMatrix) = R.λ ? R : R .= zero(eltype(R))

lrmul!(L::IdentityMatrix,
	M::AbstractMatrix,
	R::IdentityMatrix) = (L.λ && R.λ) ? M : M .= zero(eltype(M))

lrmul!(L::Diagonal,
	M::SparseMatrixCSC,
	R::IdentityMatrix) = R.λ ? lmul!(L, M) : M .= zero(eltype(M))

lrmul!(L::Diagonal,
	M::AbstractMatrix,
	R::Diagonal) = LinearAlgebra.lmul!(L, LinearAlgebra.rmul!(M, R))

lrmul!(L::Diagonal,
	M::AbstractMatrix,
	R::IdentityMatrix) = R.λ ? LinearAlgebra.lmul!(L, M) : M .= zero(eltype(M))


lrmul!(L::IdentityMatrix,
	M::AbstractMatrix,
	R::Diagonal) = L.λ ? LinearAlgebra.rmul!(M, R) : M .= zero(eltype(M))


#Julia SparseArrays dot function is very slow for Symmtric
#matrices.  See https://github.com/JuliaSparse/SparseArrays.jl/issues/83
function symdot(
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
        return dot(zero(eltype(x)), zero(eltype(A)), zero(eltype(y)))
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
