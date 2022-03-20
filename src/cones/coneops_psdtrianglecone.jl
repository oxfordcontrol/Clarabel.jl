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

    f = K.work

    # Make into matrix form for Cholesky
    # PJG: probably only necessary to fill
    # in the upper triangle (unscaled) of
    # (S,V) and wrap it in Symmetric ()
    map((M,v)->_tomat!(M,v,K),(f.S,f.Z),(s,z))

    #compute Cholesky factors
    f.cholS = cholesky(f.S, check = true)
    f.cholZ = cholesky(f.Z, check = true)
    (L1,L2) = (f.cholS.L,f.cholZ.L)

    #R is the same size as L2'*L1,
    #so use it as temporary workspace
    f.R  .= L2'*L1
    f.SVD = svd(f.R)

    #assemble  λ (diagonal), R and Rinv.
    f.λ           .= f.SVD.S
    f.Λisqrt.diag .= inv.(sqrt.(f.λ))

    f.R    .= L1*(f.SVD.V)*f.Λisqrt
    f.Rinv .= f.Λisqrt*(f.SVD.U)'*L2'

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

    R   = K.work.R
    kRR = K.work.kronRR
    B   = K.work.B
    WtW = K.work.WtW
    @inbounds kron!(kRR,R,R)

    #B .= Q'*kRR, where Q' is the svec operator
    #this could be substantially faster
    for i = 1:size(B,2)
        @views M = reshape(kRR[:,i],size(R,1),size(R,1))
        b = view(B,:,i)
        _tovec!(b,M,K)
    end

    #compute WtW = triu(B*B')
    LinearAlgebra.BLAS.syrk!('U', 'N', one(T), B, zero(T), WtW)

    pack_triu(WtWblock,WtW)

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
    for k = 1:K.n
        x[(k*(k+1)) >> 1] = K.work.λ[k]^2
    end

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

    # PJG : mayble should only really need to compute
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

  R = K.work.R

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
    Λisqrt = K.work.Λisqrt

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

# PJG: This should move to math utils, and should probably
# apply Vector -> Vector, or have a separate method to do so.
# Would also be nice to have an option that only fills in the
# upper triangle of a Symmetric matrix

#make a matrix view from a vectorized input
function _tomat!(M::AbstractMatrix{T}, x::AbstractVector{T}, K::PSDTriangleCone{T}) where {T}

    #PJG: sanity checking sizes
    @assert(K.numel == length(x))
    @assert(K.n == LinearAlgebra.checksquare(M))

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

# make an svec input from a matrix and should probably
# apply Vector -> Vector, or have a separate method to do so.
# Not clear if a another method for reading from the upper
# triangle only is useful, e.g. for a Symmetric input
function _tovec!(x::AbstractVector{T},M::AbstractMatrix{T},K::PSDTriangleCone{T}) where {T}

    #PJG: sanity checking sizes
    @assert(K.numel == length(x))
    @assert(K.n == LinearAlgebra.checksquare(M))

    ISQRT2 = 1/sqrt(T(2))

    idx = 1
    for row = 1:K.n, col = 1:row
        @inbounds x[idx] = row == col ? M[row,col] : (M[row,col]+M[col,row])*ISQRT2
        idx += 1
    end

    return nothing
end


_tomat_operator(K::PSDTriangleCone{T}) where {T} = _tomat_operator(T,K.n)

function _tomat_operator(T,n::Int)

    ISQRT2 = 1. / sqrt(T(2.))

    nrows = n^2
    ncols = (n*(n+1))>>1
    numel = nrows

    IX = zeros(Int,numel)
    JX = zeros(Int,numel)
    VX = zeros(Float64,numel)

    i = 1
    for c = 1:n, r = 1:n
        IX[i] = i
        if r == c
            JX[i] = (r*(r+1)) >> 1
            VX[i] = 1
        elseif r > c  #lower triangle
            JX[i] = ((r*(r-1)) >> 1) + c
            VX[i] = ISQRT2
        else          #upper triangle
            JX[i] = ((c*(c-1)) >> 1) + r
            VX[i] = ISQRT2
        end
        i = i+1
    end

    Q = sparse(IX,JX,VX)
end
