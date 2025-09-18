using GenericLinearAlgebra  # extends SVD, eigs etc for BigFloats

# # ----------------------------------------------------
# # Positive Semidefinite Cone
# # ----------------------------------------------------

# degree(K::PSDTriangleCone{T}) where {T} = K.n        #side dimension, M \in \mathcal{S}^{n×n}
# numel(K::PSDTriangleCone{T})  where {T} = K.numel    #number of elements

# function margins is implemented explicitly into the compositecone operation
@inline function margins_psd(
    Z::AbstractArray{T,3},
    z::AbstractVector{T},
    eigvals::AbstractMatrix{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_psd::Cint,
    αmin::T
) where {T}
    svec_to_mat_gpu!(Z, z, rng_cones, n_shift, n_psd)

    # Batched SVD decomposition
    syevjBatched!(Z, eigvals, size(eigvals,1))      #'N' returns eigenvalues only; 'V' returns both eigenvalues and eigenvectors
    αmin = min(αmin, minimum(eigvals))
    CUDA.@sync @. eigvals = max(eigvals, zero(T))
    return (αmin, sum(eigvals))
end

# place vector into sdp cone
function _kernel_scaled_unit_shift_psd!(
    z::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    psd_dim::Cint,
    n_shift::Cint,
    n_psd::Cint
) where{T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_psd
        # #adds αI to the vectorized triangle,
        # #at elements [1,3,6....n(n+1)/2]
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        @views zi = z[rng_cone_i] 
        @inbounds for k = 1:psd_dim
            zi[triangular_index(k)] += α
        end
    end

    return nothing
end

@inline function scaled_unit_shift_psd!(
    z::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    psd_dim::Cint,
    n_shift::Cint,
    n_psd::Cint
) where {T}
    kernel = @cuda launch=false _kernel_scaled_unit_shift_psd!(z, α, rng_cones, psd_dim, n_shift, n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    kernel(z, α, rng_cones, psd_dim, n_shift, n_psd; threads, blocks)
end

# unit initialization for asymmetric solves
@inline function unit_initialization_psd_gpu!(
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    psd_dim::Cint,
    n_shift::Cint,
    n_psd::Cint
 ) where{T}
 
    CUDA.@allowscalar begin
        rng = rng_cones[n_shift+1].start:rng_cones[n_shift+n_psd].stop
        @views fill!(s[rng],zero(T))
        @views fill!(z[rng],zero(T))
    end
    α = one(T)

    scaled_unit_shift_psd!(z,α,rng_cones,psd_dim,n_shift,n_psd)
    scaled_unit_shift_psd!(s,α,rng_cones,psd_dim,n_shift,n_psd)
 
    return nothing
 end

# # configure cone internals to provide W = I scaling
function _kernel_set_identity_scaling_psd!(
    R::AbstractArray{T,3},
    Rinv::AbstractArray{T,3},
    Hspsd::AbstractArray{T,3},
    psd_dim::Cint,
    n_psd::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_psd
        #Other entries of R, Rinv, Hspsd to 0's
        @inbounds for k in 1:psd_dim
            R[k,k,i] = one(T)
            Rinv[k,k,i] = one(T)
        end
        @inbounds for k in 1:triangular_number(psd_dim)     
            Hspsd[k,k,i] = one(T)
        end
    end

    return nothing
end

@inline function set_identity_scaling_psd!(
    R::AbstractArray{T,3},
    Rinv::AbstractArray{T,3},
    Hspsd::AbstractArray{T,3},
    psd_dim::Cint,
    n_psd::Cint
) where {T}
    kernel = @cuda launch=false _kernel_set_identity_scaling_psd!(R, Rinv, Hspsd, psd_dim, n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    kernel(R, Rinv, Hspsd, psd_dim, n_psd; threads, blocks)
end

@inline function update_scaling_psd!(
    L1::AbstractArray{T,3},
    L2::AbstractArray{T,3},
    U::AbstractArray{T,3}, 
    S::AbstractArray{T,2}, 
    V::AbstractArray{T,3},
    z::AbstractVector{T},
    s::AbstractVector{T},
    workmat1::AbstractArray{T,3},
    λpsd::AbstractMatrix{T},
    Λisqrt::AbstractMatrix{T},
    R::AbstractArray{T,3},
    Rinv::AbstractArray{T,3},
    Hspsd::AbstractArray{T,3},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    svec_to_mat_gpu!(L2,z,rng_cones,n_shift,n_psd)
    svec_to_mat_gpu!(L1,s,rng_cones,n_shift,n_psd)

    infoz = potrfBatched!(L2, 'L')
    infos = potrfBatched!(L1, 'L')

    # YC: This is an issue related to the batched Cholesky factorization in CUSOLVER,
    # which fills in both lower and upper triangular of each submatrix during factorization
    #Set upper triangular parts to 0
    mask_zeros!(L2, 'U')
    mask_zeros!(L1, 'U')

    # bail if the cholesky factorization fails
    if !(iszero(infoz) && iszero(infos))
        return is_scaling_success = false
    end

    #SVD of L2'*L1,
    tmp = workmat1;
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(T), L2, L1, zero(T), tmp)
    gesvdjBatched!('V', tmp, U, S, V)

    #assemble λ (diagonal), R and Rinv.
    copyto!(λpsd, S)
    CUDA.@sync @. Λisqrt = inv.(sqrt.(λpsd))

    #R = L1*(V)*Λisqrt  Λisqrt is a diagonal matrix for each psd cone
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), L1, V, zero(T), R)
    right_mul_batched!(R,Λisqrt,R)

    #Rinv = Λisqrt*(U)'*L2'
    CUDA.CUBLAS.gemm_strided_batched!('T', 'T', one(T), U, L2, zero(T), Rinv)
    left_mul_batched!(Λisqrt,Rinv,Rinv)

    #compute R*R' (upper triangular part only)
    RRt = workmat1;
    fill!(RRt, zero(T))
    CUDA.CUBLAS.gemm_strided_batched!('N', 'T', one(T), R, R, zero(T), RRt)

    #YC: RRt may not be symmetric, not sure how much effect it will be
    skron_batched!(Hspsd,RRt)
end

function _kernel_get_Hs_psd!(
    Hsblock::AbstractVector{T},
    Hs::AbstractArray{T,3},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_psd
        shift_i = i + n_shift
        rng_i = rng_blocks[shift_i]
        @views Hsi = Hs[:,:,i]
        @views Hsblocki = Hsblock[rng_i]
        
        @inbounds for j in 1:length(Hsblocki)
            Hsblocki[j] = Hsi[j]
        end
    end

    return nothing

end

@inline function get_Hs_psd!(
    Hsblocks::AbstractVector{T},
    Hspsd::AbstractArray{T,3},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    kernel = @cuda launch=false _kernel_get_Hs_psd!(Hsblocks, Hspsd, rng_blocks, n_shift, n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    kernel(Hsblocks, Hspsd, rng_blocks, n_shift, n_psd; threads, blocks)

end

# compute the product y = WᵀWx
@inline function mul_Hs_psd!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    Hspsd::AbstractArray{T,3},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_psd::Cint,
    psd_dim::Cint
) where {T}
    CUDA.@allowscalar rng = rng_cones[n_shift+1].start:rng_cones[n_shift+n_psd].stop

    #Transform it into the matrix form 
    @views tmpx = x[rng]
    @views tmpy = y[rng]

    n_tri_dim = triangular_number(psd_dim)
    n_psd_int64 = Int64(n_psd)

    X = reshape(tmpx, (n_tri_dim, n_psd_int64))
    Y = reshape(tmpy, (n_tri_dim, n_psd_int64))

    CUDA.CUBLAS.gemv_strided_batched!('N', one(T), Hspsd, X, zero(T), Y)
end

# returns ds = λ ∘ λ for the SDP cone
function _kernel_affine_ds_psd!(
    ds::AbstractVector{T},
    λpsd::AbstractMatrix{T},
    rng_cones::AbstractVector,
    psd_dim::Cint,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_psd
        #We have Λ = Diagonal(K.λ), so
        #λ ∘ λ should map to Λ.^2
        shift_idx = rng_cones[n_shift+i].start - 1
        #same as X = Λ*Λ
        @inbounds for k = 1:psd_dim
            ds[shift_idx+triangular_index(k)] = λpsd[k,i]^2
        end
    end

    return nothing
end

@inline function affine_ds_psd_gpu!(
    ds::AbstractVector{T},
    λpsd::AbstractMatrix{T},
    rng_cones::AbstractVector,
    psd_dim::Cint,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    CUDA.@allowscalar begin
        rng = rng_cones[n_shift+1].start:rng_cones[n_shift+n_psd].stop
    end
    @views fill!(ds[rng],zero(T))

    kernel = @cuda launch=false _kernel_affine_ds_psd!(ds,λpsd,rng_cones,psd_dim,n_shift,n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    kernel(ds,λpsd,rng_cones,psd_dim,n_shift,n_psd; threads, blocks)
end

@inline function combined_ds_shift_psd!(
    cones::CompositeConeGPU{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    n_shift::Cint,
    n_psd::Cint,
    σμ::T
) where {T}

    #shift vector used as workspace for a few steps 
    tmp = shift    
    R = cones.R
    Rinv = cones.Rinv
    rng_cones = cones.rng_cones
    workmat1 = cones.workmat1
    workmat2 = cones.workmat2
    workmat3 = cones.workmat3
    psd_dim = cones.psd_dim
    
    CUDA.@allowscalar begin
        rng = rng_cones[n_shift+1].start:rng_cones[n_shift+n_psd].stop
    end

     #Δz <- WΔz
    CUDA.@sync @. tmp[rng] = step_z[rng]        
    mul_Wx_psd!(step_z, tmp, R, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, false)     

    #Δs <- W⁻TΔs
    CUDA.@sync @. tmp[rng] = step_s[rng]            
    mul_WTx_psd!(step_s, tmp, Rinv, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, false)   

    #shift = W⁻¹Δs ∘ WΔz - σμe
    #X  .= (Y*Z + Z*Y)/2 
    # NB: Y and Z are both symmetric
    svec_to_mat_gpu!(workmat1,step_z,rng_cones,n_shift,n_psd)
    svec_to_mat_gpu!(workmat2,step_s,rng_cones,n_shift,n_psd)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), workmat1, workmat2, zero(T), workmat3)
    symmetric_part_gpu!(workmat3)

    mat_to_svec_gpu!(shift,workmat3,rng_cones,n_shift,n_psd)       
    
    scaled_unit_shift_psd!(shift,-σμ,rng_cones,psd_dim,n_shift,n_psd)                     

    return nothing

end


function _kernel_op_λ!(
    X::AbstractArray{T,3},
    Z::AbstractArray{T,3},
    λpsd::AbstractMatrix{T},
    psd_dim::Cint,
    n_psd::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_psd
        @views Xi = X[:,:,i]
        @views Zi = Z[:,:,i]
        @views λi = λpsd[:,i]
        for k = 1:psd_dim
            for j = 1:psd_dim
                Xi[k,j] = 2*Zi[k,j]/(λi[k] + λi[j])
            end
        end
    end

    return nothing
end

@inline function op_λ!(
    X::AbstractArray{T,3},
    Z::AbstractArray{T,3},
    λpsd::AbstractMatrix{T},
    psd_dim::Cint,
    n_psd::Cint
) where {T}

    kernel = @cuda launch=false _kernel_op_λ!(X, Z, λpsd, psd_dim, n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    CUDA.@sync kernel(X, Z, λpsd, psd_dim, n_psd; threads, blocks)
end

@inline function Δs_from_Δz_offset_psd!(
    cones::CompositeConeGPU{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T},
    n_shift::Cint,
    n_psd::Cint
) where {T}

    R = cones.R
    λpsd = cones.λpsd
    rng_cones = cones.rng_cones
    workmat1 = cones.workmat1
    workmat2 = cones.workmat2
    workmat3 = cones.workmat3
    psd_dim = cones.psd_dim

    #tmp = λ \ ds 
    svec_to_mat_gpu!(workmat2, ds, rng_cones, n_shift, n_psd)
    op_λ!(workmat1, workmat2, λpsd, psd_dim, n_psd)
    mat_to_svec_gpu!(work, workmat1, rng_cones, n_shift, n_psd) 

    #out = Wᵀ(λ \ ds) = Wᵀ(work) 
    mul_WTx_psd!(out, work, R, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, false)   

end

@inline function step_length_psd(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    Λisqrt::AbstractMatrix{T}, 
    eigvals::AbstractMatrix{T},
    d::AbstractVector{T}, 
    Rx::AbstractArray{T,3}, 
    Rinv::AbstractArray{T,3}, 
    workmat1::AbstractArray{T,3}, 
    workmat2::AbstractArray{T,3}, 
    workmat3::AbstractArray{T,3},
    αmax::T,
    rng_cones::AbstractVector,
    n_shift::Cint, 
    n_psd::Cint
) where {T}

    workΔ  = workmat1
    #d = Δz̃ = WΔz
    # We need an extra parameter since the dimension of d is not equal to that of dz
    # αz = step_length_psd_component(workΔ,d,Λisqrt,αmax)
    mul_Wx_psd!(d, dz, Rx, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, true)
    αz = step_length_psd_component_gpu(workΔ, d, Λisqrt, eigvals, n_psd, αmax)
    
    #d = Δs̃ = W^{-T}Δs
    mul_WTx_psd!(d, ds, Rinv, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, true)
    αs = step_length_psd_component_gpu(workΔ, d, Λisqrt, eigvals, n_psd, αmax)
    
    @views αmax = min(αmax,αz,αs)

    return αmax
end

function _kernel_logdet!(
    barrier::AbstractVector{T},
    fact::AbstractArray{T,3},
    psd_dim::Cint,
    n_psd::Cint
) where {T}
    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_psd
        val = zero(T)
        @inbounds for k = 1:psd_dim
            val += logsafe(fact[k,k,i])
        end
        barrier[i] = val + val
    end

    return nothing
end

@inline function _logdet_barrier_psd(
    barrier::AbstractVector{T},
    x::AbstractVector{T},
    dx::AbstractVector{T},
    alpha::T,
    workmat1::AbstractArray{T,3},
    workvec::AbstractVector{T},
    rng::UnitRange{Cint},
    psd_dim::Cint,
    n_psd::Cint
) where {T}

    Q = workmat1
    q = workvec

    CUDA.@sync @. q = x[rng] + alpha*dx[rng]
    svec_to_mat_no_shift_gpu!(Q, q, n_psd)
    info = potrfBatched!(Q, 'L')


    if iszero(info)
        kernel = @cuda launch=false _kernel_logdet!(barrier, Q, psd_dim, n_psd)
        config = launch_configuration(kernel.fun)
        threads = min(n_psd, config.threads)
        blocks = cld(n_psd, threads)
    
        CUDA.@sync kernel(barrier, Q, psd_dim, n_psd; threads, blocks)

        return sum(@view barrier[1:n_psd])
    else 
        return typemax(T)
    end

end

@inline function compute_barrier_psd(
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    workmat1::AbstractArray{T,3},
    workvec::AbstractVector{T},
    rng_cones::AbstractVector,
    psd_dim::Cint,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    CUDA.@allowscalar begin
        rng = rng_cones[n_shift+1].start:rng_cones[n_shift+n_psd].stop
    end

    barrier_d = _logdet_barrier_psd(barrier, z, dz, α, workmat1, workvec, rng, psd_dim, n_psd)
    barrier_p = _logdet_barrier_psd(barrier, s, ds, α, workmat1, workvec, rng, psd_dim, n_psd)
    return (- barrier_d - barrier_p)

end

# ---------------------------------------------
# operations supported by symmetric cones only 
# ---------------------------------------------

# implements y = Wx for the PSD cone
@inline function mul_Wx_psd!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    Rx::AbstractArray{T,3},
    rng_cones::AbstractVector,
    workmat1::AbstractArray{T,3},
    workmat2::AbstractArray{T,3},
    workmat3::AbstractArray{T,3},
    n_shift::Cint,
    n_psd::Cint,
    step_search::Bool
) where {T}

    (X,Y,tmp) = (workmat1,workmat2,workmat3)

    svec_to_mat_gpu!(X,x,rng_cones,n_shift,n_psd)

    #Y .= (R'*X*R)                #W*x
    # mul!(tmp,Rx',X)
    # mul!(Y,tmp,Rx)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(T), Rx, X, zero(T), tmp)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), tmp, Rx, zero(T), Y)

    (step_search) ? mat_to_svec_no_shift_gpu!(y,Y,n_psd) : mat_to_svec_gpu!(y,Y,rng_cones,n_shift,n_psd)

    return nothing
end

# implements y = WTx for the PSD cone
@inline function mul_WTx_psd!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    Rx::AbstractArray{T,3},
    rng_cones::AbstractVector,
    workmat1::AbstractArray{T,3},
    workmat2::AbstractArray{T,3},
    workmat3::AbstractArray{T,3},
    n_shift::Cint,
    n_psd::Cint,
    step_search::Bool
) where {T}

    (X,Y,tmp) = (workmat1,workmat2,workmat3)

    svec_to_mat_gpu!(X,x,rng_cones,n_shift,n_psd)

    #Y .= (R*X*R')                #W^T*x 
    # mul!(tmp,X,Rx')
    # mul!(Y,Rx,tmp)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'T', one(T), X, Rx, zero(T), tmp)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), Rx, tmp, zero(T), Y)

    (step_search) ? mat_to_svec_no_shift_gpu!(y,Y,n_psd) : mat_to_svec_gpu!(y,Y,rng_cones,n_shift,n_psd)

    return nothing
end

#-----------------------------------------
# internal operations for SDP cones 
# ----------------------------------------

function step_length_psd_component_gpu(
    workΔ::AbstractArray{T,3},
    d::AbstractVector{T},
    Λisqrt::AbstractMatrix{T},
    eigvals::AbstractMatrix{T},
    n_psd::Cint,
    αmax::T
) where {T}
    
    # NB: this could be made faster since we only need to populate the upper triangle 
    svec_to_mat_no_shift_gpu!(workΔ,d,n_psd)
    lrscale_psd!(workΔ,Λisqrt)
    # symmetric_part_gpu!(workΔ)

    # batched eigenvalue decomposition
    syevjBatched!(workΔ, eigvals,size(eigvals,1))

    γ = minimum(eigvals)
    if γ < 0
        return min(inv(-γ),αmax)
    else
        return αmax
    end

end

#make a matrix view from a vectorized input
function _kernel_svec_to_mat!(
    Z::AbstractArray{T,3}, 
    z::AbstractVector{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_psd
        shift_i = i + n_shift
        rng_i = rng_blocks[shift_i]
        @views Zi = Z[:,:,i]
        @views zi = z[rng_i]
        svec_to_mat!(Zi,zi)
    end

    return nothing
end

@inline function svec_to_mat_gpu!(
    Z::AbstractArray{T,3}, 
    z::AbstractVector{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_psd::Cint
) where {T}
    kernel = @cuda launch=false _kernel_svec_to_mat!(Z,z,rng_blocks,n_shift,n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    CUDA.@sync kernel(Z,z,rng_blocks,n_shift,n_psd; threads, blocks)
end

#No shift version of svec_to_mat
function _kernel_svec_to_mat_no_shift!(
    Z::AbstractArray{T,3}, 
    z::AbstractVector{T},
    n_psd::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_psd
        @views Zi = Z[:,:,i]
        dim = size(Zi,1)
        rng_i = ((i-1)*triangular_number(dim) + 1):(i*triangular_number(dim))
        @views zi = z[rng_i]
        svec_to_mat!(Zi,zi)
    end

    return nothing
end

@inline function svec_to_mat_no_shift_gpu!(
    Z::AbstractArray{T,3}, 
    z::AbstractVector{T},
    n_psd::Cint
) where {T}
    kernel = @cuda launch=false _kernel_svec_to_mat_no_shift!(Z,z,n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    CUDA.@sync kernel(Z,z,n_psd; threads, blocks)
end

#make a matrix view from a vectorized input
function _kernel_mat_to_svec!(
    z::AbstractVector{T},
    Z::AbstractArray{T,3}, 
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_psd::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_psd
        shift_i = i + n_shift
        rng_i = rng_blocks[shift_i]
        @views Zi = Z[:,:,i]
        @views zi = z[rng_i]
        mat_to_svec!(zi,Zi)
    end

    return nothing
end

@inline function mat_to_svec_gpu!(
    z::AbstractVector{T},
    Z::AbstractArray{T,3}, 
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_psd::Cint
) where {T}
    kernel = @cuda launch=false _kernel_mat_to_svec!(z,Z,rng_blocks,n_shift,n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    CUDA.@sync kernel(z,Z,rng_blocks,n_shift,n_psd; threads, blocks)
end

#No shift version of mat_to_svec
function _kernel_mat_to_svec_no_shift!(
    z::AbstractVector{T},
    Z::AbstractArray{T,3}, 
    n_psd::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_psd
        @views Zi = Z[:,:,i]
        dim = size(Zi,1)
        rng_i = ((i-1)*triangular_number(dim) + 1):(i*triangular_number(dim))
        @views zi = z[rng_i]
        mat_to_svec!(zi,Zi)
    end

    return nothing
end

@inline function mat_to_svec_no_shift_gpu!(
    z::AbstractVector{T},
    Z::AbstractArray{T,3}, 
    n_psd::Cint
) where {T}
    kernel = @cuda launch=false _kernel_mat_to_svec_no_shift!(z,Z,n_psd)
    config = launch_configuration(kernel.fun)
    threads = min(n_psd, config.threads)
    blocks = cld(n_psd, threads)

    CUDA.@sync kernel(z,Z,n_psd; threads, blocks)
end

# produce the upper triangular part of the Symmetric Kronecker product of
# a symmtric matrix A with itself, i.e. triu(A ⊗_s A) with full fill-in
function skron_full!(
    out::AbstractMatrix{T},
    A::AbstractMatrix{T},
) where {T}

    sqrt2  = sqrt(2)
    n      = size(A, 1)

    col = 1
    @inbounds for l in 1:n
        @inbounds for k in 1:l
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

                    #Also fill-in the lower triangular part
                    out[col, row] = out[row, col]

                    row += 1
                end # i
            end # j
            col += 1
        end # k
    end # l
end

function _kernel_skron!(
    out::AbstractArray{T,3}, 
    A::AbstractArray{T,3},
    n::Clong
) where {T}
    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n
        @views outi = out[:,:,i]
        @views Ai = A[:,:,i]

        skron_full!(outi,Ai)
    end

    return nothing
end

@inline function skron_batched!(
    out::AbstractArray{T,3}, 
    A::AbstractArray{T,3}
) where {T}
    n = size(out,3)

    kernel = @cuda launch=false _kernel_skron!(out,A,n)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(out,A,n; threads, blocks)
end

#right multiplication for A[:,:,i] with the diagonal matrix of B[:,i]
function _kernel_right_mul!(
    A::AbstractArray{T,3}, 
    B::AbstractArray{T,2},
    C::AbstractArray{T,3},
    n2::Cint,
    n::Cint
) where {T}
    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n
        (k,j) = divrem(i-1, n2)
        j += 1
        k += 1
        val = B[j,k]
        @inbounds for l in axes(A,1)
            C[l,j,k] = val*A[l,j,k]
        end
    end

    return nothing
end

@inline function right_mul_batched!(
    A::AbstractArray{T,3}, 
    B::AbstractArray{T,2},
    C::AbstractArray{T,3}
) where {T}
    n2 = Cint(size(A,2))
    n = n2*Cint(size(A,3))

    kernel = @cuda launch=false _kernel_right_mul!(A,B,C,n2,n)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(A,B,C,n2,n; threads, blocks)
end

#left multiplication for B[:,:,i] with the diagonal matrix of A[:,i]
function _kernel_left_mul!(
    A::AbstractArray{T,2}, 
    B::AbstractArray{T,3},
    C::AbstractArray{T,3},
    n2::Cint,
    n::Cint
) where {T}
    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n
        (k,j) = divrem(i-1, n2)
        j += 1
        k += 1
        val = A[j,k]
        @inbounds for l in axes(A,1)
            C[j,l,k] = val*B[j,l,k]
        end
    end

    return nothing
end

@inline function left_mul_batched!(
    A::AbstractArray{T,2}, 
    B::AbstractArray{T,3},
    C::AbstractArray{T,3}
) where {T}
    n2 = Cint(size(B,2))
    n = n2*Cint(size(B,3))

    kernel = @cuda launch=false _kernel_left_mul!(A,B,C,n2,n)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(A,B,C,n2,n; threads, blocks)
end