using GenericLinearAlgebra  # extends SVD, eigs etc for BigFloats

# ----------------------------------------------------
# Positive Semidefinite Cone
# ----------------------------------------------------

degree(K::PSDTriangleCone{T}) where {T} = K.n        #side dimension, M \in \mathcal{S}^{n×n}
numel(K::PSDTriangleCone{T})  where {T} = K.numel    #number of elements

function margins(
    K::PSDTriangleCone{T},
    z::AbstractVector{T},
    pd::PrimalOrDualCone
) where{T}

    if length(z) == 0
        e = T[]
        α = floatmax(T)
    else
        Z = K.data.workmat1
        svec_to_mat!(Z,z)
        e = eigvals!(Hermitian(Z))  #NB: GenericLinearAlgebra doesn't support eigvals!(::Symmetric(...))
        α = minimum(e)  #minimum eigenvalue. 
    end
    β = reduce((x,y) -> y > 0 ? x + y : x, e, init = zero(T)) # = sum(e[e.>0]) (no alloc)
    (α,β)
    
end

# place vector into sdp cone
function scaled_unit_shift!(
    K::PSDTriangleCone{T},
    z::AbstractVector{T},
    α::T,
    pd::PrimalOrDualCone
) where{T}

    #adds αI to the vectorized triangle,
    #at elements [1,3,6....n(n+1)/2]
    for k = 1:K.n
        z[triangular_index(k)] += α
    end

    return nothing
end

# unit initialization for asymmetric solves
function unit_initialization!(
    K::PSDTriangleCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
 ) where{T}
 
    s .= zero(T)
    z .= zero(T)

    #Primal or Dual doesn't matter here
    #since the cone is self dual anyway
    scaled_unit_shift!(K,s,one(T),PrimalCone)
    scaled_unit_shift!(K,z,one(T),DualCone)
 
    return nothing
 end



 function set_identity_scaling!(
    K::PSDTriangleCone{T}
) where {T}

    K.data.R    .= I(K.n)
    K.data.Rinv .= K.data.R
    K.data.Hs   .= I(size(K.data.Hs,1))

    return nothing
end

#configure cone internals to provide W = I scaling
function update_scaling!(
    K::PSDTriangleCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    if length(s) == 0 
        #bail early on zero length cone
        return true;
    end

    f = K.data

    (S,Z) = (f.workmat1,f.workmat2)
    map((M,v)->svec_to_mat!(M,v),(S,Z),(s,z))

    #compute Cholesky factors (PG: this is allocating)
    f.chol1 = cholesky!(S, check = false)
    f.chol2 = cholesky!(Z, check = false)

    # bail if the cholesky factorization fails
    if !(issuccess(f.chol1) && issuccess(f.chol2))
        return is_scaling_success = false
    end

    (L1,L2) = (f.chol1.L,f.chol2.L)

    #SVD of L2'*L1,
    tmp = f.workmat1;
    mul!(tmp,L2',L1)   
    f.SVD = svd(tmp)

    #assemble λ (diagonal), R and Rinv.
    f.λ           .= f.SVD.S
    f.Λisqrt.diag .= inv.(sqrt.(f.λ))

    #f.R = L1*(f.SVD.V)*f.Λisqrt
    mul!(f.R,L1,f.SVD.V);
    mul!(f.R,f.R,f.Λisqrt) #mul! can take R twice because Λ is diagonal

    #f.Rinv .= f.Λisqrt*(f.SVD.U)'*L2'
    mul!(f.Rinv,f.SVD.U',L2')
    mul!(f.Rinv,f.Λisqrt,f.Rinv) #mul! can take Rinv twice because Λ is diagonal

    #compute R*R' (upper triangular part only)
    RRt = f.workmat1;
    RRt .= zero(T)
    if T <: LinearAlgebra.BlasFloat
        LinearAlgebra.BLAS.syrk!('U', 'N', one(T), f.R, zero(T), RRt)
    else 
        RRt .= f.R*f.R'
    end

    # PJG: it is possibly faster to compute the whole of RRt, and not 
    # just the upper triangle using syrk!, because then skron! can be 
    # be called with a Matrix type instead of Symmetric.   The internal 
    # indexing within skron! is then more straightforward and probably 
    # faster.
    skron!(f.Hs,Symmetric(RRt))

    return is_scaling_success = true
end


function Hs_is_diagonal(
    K::PSDTriangleCone{T}
) where{T}
    return false
end


function get_Hs!(
    K::PSDTriangleCone{T},
    Hsblock::AbstractVector{T}
) where {T}

    pack_triu(Hsblock,K.data.Hs)

    return nothing
end

# compute the product y = WᵀWx
function mul_Hs!(
    K::PSDTriangleCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    work::AbstractVector{T}
) where {T}

    #PJG: using W^W here, but it maybe doesn't make 
    #sense since I have already directly computed 
    #the Hs block.   Why not just y = Symmetric(K).x, 
    #or some symv variant thereof?   
    #
    # On the other hand, it might be better to keep it this 
    # way and then *never* hold Hs in the cone work data. 
    # Perhaps the calculation of scale factors that includes 
    # the kron(R,R) onwards could be done in a more compact 
    # way since the internal Hs work variable is only really 
    # needed to populate the KKT Hs block.   For a direct 
    # method that block is never needed, so better to only 
    # form it in memory if get_Hs is actually called  
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
        ds[triangular_index(k)] = K.data.λ[k]^2
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
    work::AbstractVector{T},
    z::AbstractVector{T}
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

    Λisqrt = K.data.Λisqrt
    d      = K.data.workvec
    workΔ  = K.data.workmat1

    #d = Δz̃ = WΔz
    mul_W!(K, :N, d, dz, one(T), zero(T))
    αz = step_length_psd_component(workΔ,d,Λisqrt,αmax)

    #d = Δs̃ = W^{-T}Δs
    mul_Winv!(K, :T, d, ds, one(T), zero(T))
    αs = step_length_psd_component(workΔ,d,Λisqrt,αmax)

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

    barrier  = zero(T)
    barrier -= _logdet_barrier(K,z,dz,α)
    barrier -= _logdet_barrier(K,s,ds,α)
    return barrier 

end

function _logdet_barrier(K::PSDTriangleCone{T},x::AbstractVector{T},dx::AbstractVector{T},alpha::T) where {T}

    f = K.data
    Q = f.workmat1
    q = f.workvec 

    @. q  = x + alpha*dx
    svec_to_mat!(Q,q)

    # PG: this is allocating
    f.chol1 = cholesky!(Q, check = false)

    if issuccess(f.chol1) 
        return logdet(f.chol1)
    else 
        return typemax(T)
    end 

end

# ---------------------------------------------
# operations supported by symmetric cones only 
# ---------------------------------------------



# implements y = αWx + βy for the PSD cone
function mul_W!(
    K::PSDTriangleCone{T},
    is_transpose::Symbol,
    y::AbstractVector{T},
    x::AbstractVector{T},
    α::T,
    β::T
) where {T}

    mul_Wx_inner(
        is_transpose,
        y,x,
        α,
        β,
        K.data.R,
        K.data.workmat1,
        K.data.workmat2,
        K.data.workmat3)
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

    mul_Wx_inner(
        is_transpose,
        y,x,
        α,
        β,
        K.data.Rinv,
        K.data.workmat1,
        K.data.workmat2,
        K.data.workmat3)
    
end

# implements x = λ \ z for the SDP cone
function λ_inv_circ_op!(
    K::PSDTriangleCone{T},
    x::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    (X,Z) = (K.data.workmat1,K.data.workmat2)
    map((M,v)->svec_to_mat!(M,v),(X,Z),(x,z))

    λ = K.data.λ
    for i = 1:K.n
        for j = 1:K.n
            X[i,j] = 2*Z[i,j]/(λ[i] + λ[j])
        end
    end
    mat_to_svec!(x,X)

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

    (Y,Z) = (K.data.workmat1,K.data.workmat2)
    map((M,v)->svec_to_mat!(M,v),(Y,Z),(y,z))

    X = K.data.workmat3;

    #X  .= (Y*Z + Z*Y)/2 
    # NB: Y and Z are both symmetric
    if T <: LinearAlgebra.BlasFloat
        LinearAlgebra.BLAS.syr2k!('U', 'N', T(0.5), Y, Z, zero(T), X)
    else 
        X .= (Y*Z + Z*Y)/2
    end
    mat_to_svec!(x,Symmetric(X))

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

function mul_Wx_inner(
    is_transpose::Symbol,
    y::AbstractVector{T},
    x::AbstractVector{T},
    α::T,
    β::T,
    Rx::AbstractMatrix{T},
    workmat1::AbstractMatrix{T},
    workmat2::AbstractMatrix{T},
    workmat3::AbstractMatrix{T}
) where {T}

    (X,Y,tmp) = (workmat1,workmat2,workmat3)
    map((M,v)->svec_to_mat!(M,v),(X,Y),(x,y))

    if is_transpose === :T
        #Y .= α*(R*X*R')                #W^T*x    or....
        # Y .= α*(Rinv*X*Rinv') + βY    #W^{-T}*x
        mul!(tmp,X,Rx',one(T),zero(T))
        mul!(Y,Rx,tmp,α,β)
    else  # :N
        #Y .= α*(R'*X*R)                #W*x       or...
        # Y .= α*(Rinv'*X*Rinv) + βY    #W^{-1}*x
        mul!(tmp,Rx',X,one(T),zero(T))
        mul!(Y,tmp,Rx,α,β)
    end

    mat_to_svec!(y,Y)

    return nothing
end

function step_length_psd_component(
    workΔ::Matrix{T},
    d::Vector{T},
    Λisqrt::Diagonal{T},
    αmax::T
) where {T}

    if length(d) == 0
        γ = floatmax(T)
    else 
        # NB: this could be made faster since we only need to populate the upper triangle 
        svec_to_mat!(workΔ,d)
        lrscale!(Λisqrt.diag,workΔ,Λisqrt.diag)
        # GenericLinearAlgebra doesn't support eigvals!(::Symmetric(::Matrix)), 
        # and doesn't support choosing a subset of values 
        if T <: LinearAlgebra.BlasFloat
            γ = eigvals!(Hermitian(workΔ),1:1)[1] #minimum eigenvalue
        else 
            γ = eigvals!(Hermitian(workΔ))[1] #minimum eigenvalue, a bit slower
        end
    end

    if γ < 0
        return min(inv(-γ),αmax)
    else
        return αmax
    end

end

#make a matrix view from a vectorized input
function svec_to_mat!( M::AbstractMatrix{T}, x::AbstractVector{T}) where {T}

    ISQRT2 = inv(sqrt(T(2)))

    idx = 1
    for col in axes(M,2), row in 1:col
        if row == col
            M[row,col] = x[idx]
        else
            M[row,col] = x[idx]*ISQRT2
            M[col,row] = x[idx]*ISQRT2
        end
        idx += 1
    end
end


function mat_to_svec!(x::AbstractVector{T},M::AbstractMatrix{T}) where {T}

    ISQRT2 = 1/sqrt(T(2))

    idx = 1
    for col in axes(M,2), row in 1:col
        @inbounds x[idx] = row == col ? M[row,col] : (M[row,col]+M[col,row])*ISQRT2
        idx += 1
    end

    return nothing
end


# produce the upper triangular part of the Symmetric Kronecker product of
# a symmtric matrix A with itself, i.e. triu(A ⊗_s A)
function skron!(
    out::Matrix{T},
    A::Symmetric{T, Matrix{T}},
) where {T}

    sqrt2  = sqrt(2)
    n      = size(A, 1)

    col = 1
    for l in 1:n
        for k in 1:l
            row = 1
            kl_eq = k == l

            @inbounds for j in 1:n
                Ajl = A[j, l]
                Ajk = A[j, k]

                @inbounds for i in 1:j
                    (row > col) && break
                    ij_eq = i == j

                    if (ij_eq, kl_eq) == (false, false)
                        out[row, col] = A[i, k] * Ajl + A[i, l] * Ajk
                    elseif (ij_eq, kl_eq) == (true, false) 
                        out[row, col] = sqrt2 * Ajl * Ajk
                    elseif (ij_eq, kl_eq) == (false, true)  
                        out[row, col] = sqrt2 * A[i, l] * Ajk
                    else (ij_eq,kl_eq) == (true, true)
                        out[row, col] = Ajl * Ajl
                    end 

                    row += 1
                end # i
            end # j
            col += 1
        end # k
    end # l
end