# ----------------------------------------------------
# Positive Semidefinite Cone
# ----------------------------------------------------

numel(K::PSDTriangleCone{T})  where {T} = K.numel    #number of elements
degree(K::PSDTriangleCone{T}) where {T} = K.n        #side dimension, M \in \mathcal{S}^{n×n}


# compute the maximum step that shifts vector into socone
function max_shift_step!(
    K::PSDTriangleCone{T},
    z::AbstractVector{T}
) where{T}
    Z = K.work.workmat1
    _svec_to_mat!(Z,z,K)

    α = eigvals(Symmetric(Z),1:1)[1]  #min eigenvalue

    return -α
    
end

# place vector into sdp cone
function shift_to_cone!(
    K::PSDTriangleCone{T},
    z::AbstractVector{T},
    α::T
) where{T}

    Z = K.work.workmat1
    _svec_to_mat!(Z,z,K)

    add_scaled_e!(K,z,α)

    return nothing
end

# unit initialization for asymmetric solves
function unit_initialization!(
    K::PSDTriangleCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
 ) where{T}
 
     z .= zero(T)
     s .= zero(T)
     add_scaled_e!(K,z,one(T))
     add_scaled_e!(K,s,one(T))
 
    return nothing
 end






 function set_identity_scaling!(
    K::PSDTriangleCone{T}
) where {T}

    K.work.R    .= I(K.n)
    K.work.Rinv .= K.work.R

    return nothing
end

function Hs_is_diagonal(
    K::PSDTriangleCone{T}
) where{T}
    return false
end

#configure cone internals to provide W = I scaling
function update_scaling!(
    K::PSDTriangleCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
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

    # PJG : allocating 
    f.R    .= L1*(f.SVD.V)*f.Λisqrt
    f.Rinv .= f.Λisqrt*(f.SVD.U)'*L2'

    return nothing
end

function get_Hs!(
    K::PSDTriangleCone{T},
    Hsblock::AbstractVector{T}
) where {T}

    # we should return here the upper triangular part
    # of the matrix Q* (RR^T) ⨂ (RR^T) * P.  The operator
    # P is a matrix that transforms a packed triangle to
    # a vectorized full matrix.

    R   = K.work.R
    kRR = K.work.kronRR
    B   = K.work.B
    Hs = K.work.Hs
    @inbounds kron!(kRR,R,R)

    #B .= Q'*kRR, where Q' is the svec operator
    #this could be substantially faster
    for i = 1:size(B,2)
        @views M = reshape(kRR[:,i],size(R,1),size(R,1))
        b = view(B,:,i)
        _mat_to_svec!(b,M,K)
    end

    #compute Hs = triu(B*B')
    LinearAlgebra.BLAS.syrk!('U', 'N', one(T), B, zero(T), Hs)

    _pack_triu(Hsblock,Hs)

    return nothing
end

# compute the product y = WᵀWx
function mul_Hs!(
    K::PSDTriangleCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    work::AbstractVector{T}
) where {T}

    mul_W!(K,:N,work,x,one(T),zero(T))    #work = Wx
    mul_W!(K,:T,y,work,one(T),zero(T))    #y = Wᵀwork = W^TWx

end

# returns ds = λ ∘ λ for the SDP cone
function affine_ds!(
    K::PSDTriangleCone{T},
    ds::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

    #We have Λ = Diagonal(K.λ), so
    #λ ∘ λ should map to Λ.^2
    ds .= zero(T)

    #same as X = Λ*Λ
    for k = 1:K.n
        ds[(k*(k+1)) >> 1] = K.work.λ[k]^2
    end

end

function combined_ds_shift!(
    K::PSDTriangleCone{T},
    dz::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}

    _combined_ds_shift_symmetric!(K,dz,step_z,step_s,σμ);
end

function Δs_from_Δz_offset!(
    K::PSDTriangleCone{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T}
) where {T}

    _Δs_from_Δz_offset_symmetric!(K,out,ds,work);
end

##return maximum allowable step length while remaining in the psd cone
function step_length(
    K::PSDTriangleCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     settings::Settings{T},
     αmax::T
) where {T}

    Λisqrt = K.work.Λisqrt
    d   = K.work.workvec

    #d = Δz̃ = WΔz
    mul_W!(K, :N, d, dz, one(T), zero(T))
    αz = _step_length_psd_component(K,d,Λisqrt,αmax)

    #d = Δs̃ = W^{-T}Δs
    mul_Winv!(K, :T, d, ds, one(T), zero(T))
    αs = _step_length_psd_component(K,d,Λisqrt,αmax)

    return (αz,αs)
end

function compute_barrier(
    K::PSDTriangleCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T
) where {T}

    # We should return this, but in a smarter way.
    # This is not yet implemented, but would only 
    # be required for problems mixing PSD and 
    # asymmetric cones 
    # 
    # return -log(det(s)) - log(det(z))

    error("Mixed PSD and Exponential/Power cones are not yet supported")

end

# ---------------------------------------------
# operations supported by symmetric cones only 
# ---------------------------------------------

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

# implements y = αWx + βy for the PSD cone
function mul_W!(
    K::PSDTriangleCone{T},
    is_transpose::Symbol,
    y::AbstractVector{T},
    x::AbstractVector{T},
    α::T,
    β::T
) where {T}

  β == 0 ? y .= 0 : y .*= β

  (X,Y) = (K.work.workmat1,K.work.workmat2)
  map((M,v)->_svec_to_mat!(M,v,K),(X,Y),(x,y))

  R = K.work.R

  # PJG : allocating 
  if is_transpose === :T
      Y .+= α*(R*X*R')  #W^T*x
  else  # :N
      Y .+= α*(R'*X*R)  #W*x
  end

  _mat_to_svec!(y,Y,K)

  return nothing
end

# implements y = αW^{-1}x + βy for the psd cone
function mul_Winv!(
    K::PSDTriangleCone{T},
    is_transpose::Symbol,
    y::AbstractVector{T},
    x::AbstractVector{T},
    α::T,
    β::T
) where {T}

    β == 0 ? y .= 0 : y .*= β

    (X,Y) = (K.work.workmat1,K.work.workmat2)
    map((M,v)->_svec_to_mat!(M,v,K),(X,Y),(x,y))

    Rinv = K.work.Rinv

    # PJG : allocating 
    if is_transpose === :T
        Y .+= α*(Rinv*X*Rinv')  #W^{-T}*x
    else # :N
        Y .+= α*(Rinv'*X*Rinv)  #W^{-1}*x
    end

    _mat_to_svec!(y,Y,K)

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

# ---------------------------------------------
# Jordan algebra operations for symmetric cones 
# ---------------------------------------------

# implements x = y ∘ z for the SDP cone
function circ_op!(
    K::PSDTriangleCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    (Y,Z) = (K.work.workmat1,K.work.workmat2)
    map((M,v)->_svec_to_mat!(M,v,K),(Y,Z),(y,z))

    # PJG : allocating 
    Y  .= (Y*Z + Z*Y)/2
    _mat_to_svec!(x,Y,K)

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

#-----------------------------------------
# internal operations for SDP cones 
# ----------------------------------------

function _step_length_psd_component(
    K::PSDTriangleCone{T},
    d::Vector{T},
    Λisqrt::Diagonal{T},
    αmax::T
) where {T}

    Δ = K.work.workmat1
    _svec_to_mat!(Δ,d,K)

    # NB: this could be made faster since 
    # we only need to populate the upper 
    # triangle 
    lrscale!(Λisqrt.diag,Δ,Λisqrt.diag)
    M = Symmetric(Δ)

    γ = eigvals(M,1:1)[1] #minimum eigenvalue
    if γ < 0
        return min(inv(-γ),αmax)
    else
        return αmax
    end

end


