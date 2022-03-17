# ----------------------------------------------------
# Positive Semidefinite Cone
# ----------------------------------------------------

numel(K::PSDTriangleCone{T})  where {T} = K.numel    #number of elements
degree(K::PSDTriangleCone{T}) where {T} = K.n        #side dimension, M \in \mathcal{S}^{n×n}


#PSD cone returns a dense WtW block
function WtW_is_diagonal(
    K::PSDTriangleCone{T}
) where{T}
    return false
end

function update_scaling!(
    K::PSDTriangleCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
) where {T}

    #PJG: allocation here.   Remove
    (S,Z) = map(m->zeros(K.n,K.n), (s,z))
    map((M,v)->_tomat!(M,v,K),(S,Z),(s,z))

    f = K.work

    #compute Cholesky factors
    f.cholS = cholesky(S, check = true)
    f.cholZ = cholesky(Z, check = true)

    #explicit factors
    f.L1    = f.cholS.L
    f.L2    = f.cholZ.L

    #product L2'L1, hugely wasteful of memory here
    M = f.L2'*f.L1

    #SVD of M.   Requires further workspace
    f.SVD = svd(M)

    #explicit extraction of factors.  Maybe not needed
    f.U = f.SVD.U
    f.λ = f.SVD.S
    f.V = f.SVD.V

    #assemble R and Rinv.   Maybe not needed
    isqrtλ = Diagonal(inv.(sqrt.(f.λ)))
    f.R    = f.L1*f.V*isqrtλ
    f.Rinv = isqrtλ*f.U'*f.L2'

    return nothing
end



#configure cone internals to provide W = I scaling
function set_identity_scaling!(
    K::PSDTriangleCone{T}
) where {T}

    K.work.R    .= I(K.n)
    K.work.Rinv .= K.work.R

    return nothing
end

function get_WtW_block!(
    K::PSDTriangleCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    # we should return here the upper triangular part
    # of the matrix Q* (RR^T) ⨂ (RR^T) * P.  The operator
    # P is a matrix that transforms a packed triangle to
    # a vectorized full matrix.  Q does the opposite.
    # This is crazily inefficient and needs a rewrite

    R = K.work.R
    RRt = R*R'

    #we we need to right multiply by mapping tht takes
    #up from the PSDtriangle to the vectorized full matrix,
    #and left multiply by the mapping that takes us back
    P   = _tovec_operator(K)
    Q   = _tomat_operator(K)
    WtW = P*kron(RRt, RRt)*Q

    vec = triu_as_vector(WtW)
    WtWblock .= vec

    return nothing
end

# returns x = λ ∘ λ for the SDP cone
function λ_circ_λ!(
    K::PSDTriangleCone{T},
    x::AbstractVector{T}
) where {T}

    #We have Λ = Diagonal(K.λ), so
    #λ ∘ λ should map to Λ.^2
    x .= zero(T)

    #same as X = Λ*Λ
    x[1:(K.n+1):end] .= K.work.λ.^2

end

# implements x = y ∘ z for the SDP cone
#PJG Bottom p5, CVXOPT
function circ_op!(
    K::PSDTriangleCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    #PJG: allocation here.   Remove
    (X,Y,Z) = map(m->zeros(K.n,K.n), (x,y,z))
    map((M,v)->_tomat!(M,v,K),(X,Y,Z),(x,y,z))

    X  .= Y*Z + Z*Y
    X .*= 0.5

    _tovec!(x,X,K)

    return nothing
end

# implements x = λ \ z for the SDP cone
# PJG, Top page 14, \S5, CVXOPT
function λ_inv_circ_op!(
    K::PSDTriangleCone{T},
    x::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    #PJG: allocation here.   Remove
    (X,Z) = map(m->zeros(K.n,K.n), (x,z))
    map((M,v)->_tomat!(M,v,K),(X,Z),(x,z))

    # PJG : should only really need to compute
    # a triangular part of this matrix.  Keeping
    # like this for now until something works
    λ = K.work.λ
    for i = 1:K.n
        for j = 1:K.n
            X[i,j] = 2*Z[i,j]/(λ[i] + λ[j])
        end
    end
    _tovec!(x,X,K)

    return nothing
end

# implements x = y \ z for the SDP cone
# PJG, Top page 14, \S5, CVXOPT
function inv_circ_op!(
    K::PSDTriangleCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    #PJG: allocation here.   Remove
    (X,S,Z) = map(m->zeros(K.n,K.n), (x,s,z))
    map((M,v)->_tomat!(M,v,K),(X,S,Z),(x,s,z))

    # X should be the solution to (YX + XY)/2 = Z

    # PJG: or general arguments this requires solution to a symmetric
    # Sylvester equation.  Throwing an error here since I do not think
    # the inverse of the ∘ operator is ever required for general arguments,
    # and solving this equation is best avoided.

    error("This function not implemented and should never be reached.")

    _tovec!(x,X,K)

    return nothing
end

# place vector into SDP cone

function shift_to_cone!(
    K::PSDTriangleCone{T},
    z::AbstractVector{T}
) where{T}

    Z = zeros(K.n,K.n)
    _tomat!(Z,z,K)

    α = eigvals(Symmetric(Z),1:1)[1]  #min eigenvalue

    if(α < eps(T))
        #done in two stages since otherwise (1-α) = -α for
        #large α, which makes z exactly 0. (or worse, -0.0 )
        add_scaled_e!(K,z,-α)
        add_scaled_e!(K,z,one(T))
    end

    return nothing
end


# implements y = αWx + βy for the PSD cone
function gemv_W!(
    K::PSDTriangleCone{T},
    is_transpose::Symbol,
    x::AbstractVector{T},
    y::AbstractVector{T},
    α::T,
    β::T
) where {T}

  β == 0 ? y .= 0 : y .*= β

  #PJG :allocated.  Probably some tomats and
  #multiplies can be avoided if \beta = 0 or \alpha 1
  (X,Y) = map(m->zeros(K.n,K.n), (x,y))
  map((M,v)->_tomat!(M,v,K),(X,Y),(x,y))

  R   = K.work.R

  #PJG: needs unit test since only one of these
  #cases is explicitly described in the CVXOPT paper
  if is_transpose === :T
      Y .+= α*(R*X*R')  #W^T*x
  else  # :N
      Y .+= α*(R'*X*R)  #W*x
  end

  _tovec!(y,Y,K)

  return nothing
end

# implements y = αW^{-1}x + βy for the psd cone
function gemv_Winv!(
    K::PSDTriangleCone{T},
    is_transpose::Symbol,
    x::AbstractVector{T},
    y::AbstractVector{T},
    α::T,
    β::T
) where {T}

    β == 0 ? y .= 0 : y .*= β

    #PJG :allocated.  Probably some tomats and
    #multiplies can be avoided if \beta = 0 or \alpha 1
    (X,Y) = map(m->zeros(K.n,K.n), (x,y))
    map((M,v)->_tomat!(M,v,K),(X,Y),(x,y))

    Rinv = K.work.Rinv

    #PJG: needs unit test since only one of these
    #cases is explicitly described in the CVXOPT paper
    if is_transpose === :T
        Y .+= α*(Rinv*X*Rinv')  #W^{-T}*x
    else # :N
        Y .+= α*(Rinv'*X*Rinv)  #W^{-1}*x
    end

    _tovec!(y,Y,K)

    return nothing
end


# implements y = (W^TW)^{-1}x
function mul_WtWinv!(
    K::PSDTriangleCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T}
) where {T}

    #PJG: needs unit test?   Aliasing not allowed
    #Also check aliasing in other cones, esp. SOC
    #PJG: Here it seems wasteful to scale/unscale
    #into/out of scaled matrix form twice
    gemv_Winv!(K,:T,y,y,one(T),zero(T))
    gemv_Winv!(K,:N,x,y,one(T),zero(T))

    return nothing
end

# implements y = W^TW^x
function mul_WtW!(
    K::PSDTriangleCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T}
) where {T}

    #PJG: Here it seems wasteful to scale/unscale
    #into/out of scaled matrix form twice
    gemv_W!(K,:N,y,y,one(T),zero(T))
    gemv_W!(K,:T,x,y,one(T),zero(T))

    return nothing
end

# implements y = y + αe for the SDP cone
function add_scaled_e!(
    K::PSDTriangleCone{T},
    x::AbstractVector{T},
    α::T
) where {T}

    #adds αI to the vectorized triangle,
    #at elements [1,3,6....n(n+1)/2]
    for k = 1:K.n
        x[(k*(k+1))>>1] += α
    end

    return nothing
end


##return maximum allowable step length while remaining in the psd cone
function step_length(
    K::PSDTriangleCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T}
) where {T}

    #PJG: this inv sqrt is repeating
    Λisqrt = Diagonal(inv.(sqrt.(K.work.λ)))

    #PJG: DEBUG: allocs here requires removal
    d   = similar(dz)

    #d = Δz̃ = WΔz
    gemv_W!(K, :N, dz, d, one(T), zero(T))
    αz = _step_length_psd_component(K,d,Λisqrt)

    #d = Δs̃ = W^{-T}Δs
    gemv_Winv!(K, :T, ds, d, one(T), zero(T))
    αs = _step_length_psd_component(K,d,Λisqrt)

    return (αz,αs)
end


function _step_length_psd_component(
    K::PSDTriangleCone,
    d::Vector{T},
    Λisqrt::Diagonal{T}
) where {T}

    #PJG:passing K since it probably need a workspace

    #allocate.   Slow AF
    Δ = zeros(K.n,K.n)
    _tomat!(Δ,d,K)

    #allocate.   Slow AF
    M = Symmetric(Λisqrt*Δ*Λisqrt)

    γ = eigvals(M,1:1)[1] #minimum eigenvalue
    α = γ < 0 ? inv(-γ) : inv(eps(T))
    return α

end

# -------------------
# internal utilities for this cone
#--------------------

#make a matrix view from a vectorized input
function _tomat!(M::AbstractMatrix{T}, x::AbstractVector{T}, K::PSDTriangleCone{T}) where {T}

    #PJG: sanity checking sizes
    @assert(K.numel == length(x))
    @assert(K.n     == LinearAlgebra.checksquare(M))

    ISQRT2 = inv(sqrt(T(2)))

    #PJG: I am filling in the whole thing, not just the upper
    #triangular part.   For Cholesky etc only the upper triangle (scaled)
    #is needed probably
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

#make a matrix view from a vectorized input
function _tovec!(x::AbstractVector{T},M::AbstractMatrix{T},K::PSDTriangleCone{T}) where {T}

    #PJG: sanity checking sizes
    @assert(K.numel == length(x))
    @assert(K.n     == LinearAlgebra.checksquare(M))

    SQRT2 = sqrt(T(2))

    #PJG: I am reading out only the upper triangle, even though (??)
    #the matrix being passed could be filled in (and, I hope, symmetric)

    idx = 1
    for col = 1:K.n, row = 1:col
        x[idx] = row == col ? M[row,col] : M[row,col]*SQRT2
        idx += 1
    end

end

#This function needs a rewrite, or to be done
#in a different way
function _tovec_operator(K::PSDTriangleCone{T}) where {T}

    #sqrt 2 a problem here since I don't know the type

    n = K.n
    D = triu(ones(n,n))*sqrt(2)
    D = D - Diagonal(D) + I(n)
    dD = Diagonal(D[:])
    Q = dD[findall(D[:] .!= 0),:]
    return Q

end


#This function needs a rewrite, or to be done
#in a different way
function _tomat_operator(K::PSDTriangleCone{T}) where {T}

    #sqrt 2 a problem here since I don't know the type

    n = K.n
    S = ones(n,n)*(1/sqrt(2))
    S= S - Diagonal(S) + I(n)

    A = triu(ones(n,n))
    A[:] = cumsum(A[:]) .* A[:]
    A = copy(Symmetric(A,:U))

    rows = collect(1:n^2)
    cols = A[:]
    vals = S[:]

    P = sparse(rows,cols,vals)

    return P

end
