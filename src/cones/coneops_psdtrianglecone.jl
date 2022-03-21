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

    (S,Z) = (f.workmat1,f.workmat2)
    map((M,v)->_svec_to_mat!(M,v,K),(S,Z),(s,z))

    #compute Cholesky factors
    f.cholS = cholesky!(S, check = true)
    f.cholZ = cholesky!(Z, check = true)
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
        _mat_to_svec!(b,M,K)
    end

    #compute WtW = triu(B*B')
    LinearAlgebra.BLAS.syrk!('U', 'N', one(T), B, zero(T), WtW)

    _pack_triu(WtWblock,WtW)

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
function circ_op!(
    K::PSDTriangleCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    (Y,Z) = (K.work.workmat1,K.work.workmat2)
    map((M,v)->_svec_to_mat!(M,v,K),(Y,Z),(y,z))

    Y  .= (Y*Z + Z*Y)/2
    _mat_to_svec!(x,Y,K)

    return nothing
end

# implements x = λ \ z for the SDP cone
function λ_inv_circ_op!(
    K::PSDTriangleCone{T},
    x::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    (X,Z) = (K.work.workmat1,K.work.workmat2)
    map((M,v)->_svec_to_mat!(M,v,K),(X,Z),(x,z))

    λ = K.work.λ
    for i = 1:K.n
        for j = 1:K.n
            X[i,j] = 2*Z[i,j]/(λ[i] + λ[j])
        end
    end
    _mat_to_svec!(x,X,K)

    return nothing
end

# implements x = y \ z for the SDP cone
function inv_circ_op!(
    K::PSDTriangleCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    # X should be the solution to (YX + XY)/2 = Z

    #  For general arguments this requires solution to a symmetric
    # Sylvester equation.  Throwing an error here since I do not think
    # the inverse of the ∘ operator is ever required for general arguments,
    # and solving this equation is best avoided.

    error("This function not implemented and should never be reached.")

    return nothing
end

# place vector into SDP cone

function shift_to_cone!(
    K::PSDTriangleCone{T},
    z::AbstractVector{T}
) where{T}

    Z = K.work.workmat1
    _svec_to_mat!(Z,z,K)

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

  (X,Y) = (K.work.workmat1,K.work.workmat2)
  map((M,v)->_svec_to_mat!(M,v,K),(X,Y),(x,y))

  R = K.work.R

  if is_transpose === :T
      Y .+= α*(R*X*R')  #W^T*x
  else  # :N
      Y .+= α*(R'*X*R)  #W*x
  end

  _mat_to_svec!(y,Y,K)

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

    (X,Y) = (K.work.workmat1,K.work.workmat2)
    map((M,v)->_svec_to_mat!(M,v,K),(X,Y),(x,y))

    Rinv = K.work.Rinv

    if is_transpose === :T
        Y .+= α*(Rinv*X*Rinv')  #W^{-T}*x
    else # :N
        Y .+= α*(Rinv'*X*Rinv)  #W^{-1}*x
    end

    _mat_to_svec!(y,Y,K)

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

    Λisqrt = K.work.Λisqrt
    d   = K.work.workvec

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

    Δ = K.work.workmat1
    _svec_to_mat!(Δ,d,K)

    #allocate.   Slow AF
    M = Symmetric(Λisqrt*Δ*Λisqrt)

    γ = eigvals(M,1:1)[1] #minimum eigenvalue
    α = γ < 0 ? inv(-γ) : inv(eps(T))
    return α

end
