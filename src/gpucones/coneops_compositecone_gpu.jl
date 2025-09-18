
degree(cones::CompositeConeGPU{T}) where {T} = cones.degree
numel(cones::CompositeConeGPU{T}) where {T}  = cones.numel

# # -----------------------------------------------------
# # dispatch operators for multiple cones
# # -----------------------------------------------------

function is_symmetric(cones::CompositeConeGPU{T}) where {T}
    #true if all pieces are symmetric.  
    #determined during obj construction
    return cones._is_symmetric
end

function allows_primal_dual_scaling(cones::CompositeConeGPU{T}) where {T}
    # always true for the GPU settings
    return true                         #
end


NVTX.@annotate "rectify_equilibration!" function rectify_equilibration!(
    cones::CompositeConeGPU{T},
     δ::CuVector{T},
     e::CuVector{T}
) where{T}

    n_rec = cones.n_soc + cones.n_exp + cones.n_pow + cones.n_psd
    any_changed = (n_rec > 0) ? true : false

    #we will update e <- δ .* e using return values
    #from this function.  default is to do nothing at all
    CUDA.@sync @. δ = one(T)

    if any_changed
        n_shift = cones.n_linear
        rectify_equilibration_gpu!(δ, e, cones.rng_cones, n_shift, n_rec)
    end

    return any_changed
end

NVTX.@annotate "margins" function margins(
    cones::CompositeConeGPU{T},
    z::CuVector{T},
    pd::PrimalOrDualCone,
) where {T}
    αmin = typemax(T)
    β = zero(T)

    n_linear = cones.n_linear
    n_nn  = cones.n_nn
    n_soc = cones.n_soc
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    idx_inq = cones.idx_inq
    Z = cones.workmat1
    eigvals = cones.eigvals

    α = cones.α
    @. α = αmin
    
    if n_nn > 0
        (αmin,val) = margins_nonnegative(z, α, rng_cones, idx_inq, αmin)
        β += val
    end

    if n_soc > 0
        n_shift = n_linear
        (αmin,val) = margins_soc(z, α, rng_cones, n_shift, n_soc, αmin)
        β += val
    end

    if n_psd > 0
        n_shift = n_linear + n_soc
        (αmin,val) = margins_psd(Z, z, eigvals, rng_cones, n_shift, n_psd, αmin)
        β += val
    end

    return (αmin,β)
end

NVTX.@annotate "scaled_unit_shift" function scaled_unit_shift!(
    cones::CompositeConeGPU{T},
    z::CuVector{T},
    α::T,
    pd::PrimalOrDualCone
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq

    scaled_unit_shift_zero!(z, rng_cones, idx_eq, pd)

    scaled_unit_shift_nonnegative!(z, rng_cones, idx_inq, α)

    if n_soc > 0
        n_shift = n_linear
        scaled_unit_shift_soc!(z, rng_cones, α, n_shift, n_soc)
    end

    if n_psd > 0
        n_shift = n_linear + n_soc
        scaled_unit_shift_psd!(z, α, rng_cones, psd_dim, n_shift, n_psd)
    end

    synchronize()
    return nothing

end

# unit initialization for asymmetric solves
NVTX.@annotate "unit_initialization" function unit_initialization!(
    cones::CompositeConeGPU{T},
    z::CuVector{T},
    s::CuVector{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim

    αp = cones.αp
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq

    unit_initialization_zero!(z, s, rng_cones, idx_eq)

    unit_initialization_nonnegative!(z, s, rng_cones, idx_inq)
    
    if n_soc > 0
        n_shift = n_linear
        unit_initialization_soc!(z, s, rng_cones, n_shift, n_soc)
    end

    if n_exp > 0
        n_shift = n_linear+n_soc
        unit_initialization_exp!(z, s, rng_cones, n_shift, n_exp)
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        unit_initialization_pow!(z, s, αp, rng_cones, n_shift, n_pow)
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        unit_initialization_psd_gpu!(z,s,rng_cones,psd_dim,n_shift,n_psd)
    end

    synchronize()
    return nothing

end

NVTX.@annotate "set_identity_scaling!" function set_identity_scaling!(
    cones::CompositeConeGPU{T}
) where {T}

    n_linear    = cones.n_linear
    n_soc       = cones.n_soc
    n_psd       = cones.n_psd
    rng_cones   = cones.rng_cones
    w           = cones.w
    
    set_identity_scaling_nonnegative!(w, rng_cones, cones.idx_inq)

    if n_soc > 0
        set_identity_scaling_soc!(w, cones.η, rng_cones, n_linear, n_soc)
        set_identity_scaling_soc_sparse!(cones.d, cones.vut, rng_cones, n_linear, cones.n_sparse_soc)
    end

    if n_psd > 0
        set_identity_scaling_psd!(cones.R, cones.Rinv, cones.Hspsd, cones.psd_dim, n_psd)
    end

    synchronize()
    return nothing
end

NVTX.@annotate "update_scaling!" function update_scaling!(
    cones::CompositeConeGPU{T},
    s::CuVector{T},
    z::CuVector{T},
	μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_sparse_soc = cones.n_sparse_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    αp = cones.αp
    grad = cones.grad
    Hs = cones.Hs
    H_dual = cones.H_dual
    rng_cones = cones.rng_cones
    idx_inq = cones.idx_inq
    w = cones.w
    λ = cones.λ
    η = cones.η

    d = cones.d
    vut = cones.vut
    numel_linear = cones.numel_linear

    update_scaling_nonnegative!(s, z, w, λ, rng_cones, idx_inq)

    n_shift = n_linear

    if n_sparse_soc > 0 && n_sparse_soc <= SPARSE_SOC_PARALELL_NUM
        update_scaling_soc_parallel_medium!(s, z, w, λ, η, 
                                            cones.worksoc1, cones.worksoc2, cones.worksoc3, 
                                            rng_cones, n_shift, n_sparse_soc)
        n_shift += n_sparse_soc
        n_soc -= n_sparse_soc       
    end

    if n_soc > 0
        update_scaling_soc!(s, z, w, λ, η, rng_cones, n_shift, n_soc)
        n_shift += n_soc
    end  

    # off-diagonal update
    if n_sparse_soc > SPARSE_SOC_PARALELL_NUM
        update_scaling_soc_sparse_parallel!(w, η, d, vut, rng_cones, numel_linear, n_linear, n_sparse_soc)
    elseif n_sparse_soc > 0
        update_scaling_soc_sparse_parallel_medium!(w, η, d, vut, rng_cones, numel_linear, n_linear, n_sparse_soc)
    end

    if n_exp > 0
        update_scaling_exp!(s, z, grad, Hs, H_dual, rng_cones, μ, scaling_strategy, n_shift, n_exp)
        n_shift += n_exp
    end

    if n_pow > 0
        update_scaling_pow!(s, z, grad, Hs, H_dual, αp, rng_cones, μ, scaling_strategy, n_shift, n_exp, n_pow)
        n_shift += n_pow
    end

    if n_psd > 0
        update_scaling_psd!(cones.chol1, cones.chol2, cones.U, cones.S, cones.V, z, s, cones.workmat1, cones.λpsd, cones.Λisqrt, cones.R, cones.Rinv, cones.Hspsd, rng_cones, n_shift, n_psd)
    end

    synchronize()
    return is_scaling_success = true
end

# Update Hs block for each cone.
NVTX.@annotate "get_Hs!" function get_Hs!(
    cones::CompositeConeGPU{T},
    Hsblocks::CuVector{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_dense_soc = cones.n_dense_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    Hs = cones.Hs
    Hspsd = cones.Hspsd
    rng_blocks = cones.rng_blocks
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq
    w = cones.w
    η = cones.η

    #sparse SOCs
    n_sparse_soc = cones.n_sparse_soc
    d = cones.d

    get_Hs_zero!(Hsblocks, rng_blocks, idx_eq)

    get_Hs_nonnegative!(Hsblocks, w, rng_cones, rng_blocks, idx_inq)

    if n_soc > 0
        n_shift = n_linear
        if n_sparse_soc > SPARSE_SOC_PARALELL_NUM
            get_Hs_soc_sparse_parallel!(Hsblocks, η, d, rng_blocks, n_shift, n_sparse_soc)
        elseif n_sparse_soc > 0
            get_Hs_soc_sparse_parallel_medium!(Hsblocks, η, d, rng_blocks, n_shift, n_sparse_soc)
        end

        if n_dense_soc > 0 
            get_Hs_soc_dense!(Hsblocks, w, η, rng_cones, rng_blocks, n_shift + n_sparse_soc, n_sparse_soc, n_dense_soc)
        end
    end

    if n_exp > 0
        n_shift = n_linear+n_soc
        get_Hs_exp!(Hsblocks, Hs, rng_blocks, n_shift, n_exp)
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        get_Hs_pow!(Hsblocks, Hs, rng_blocks, n_shift, n_exp, n_pow)
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        get_Hs_psd!(Hsblocks, Hspsd, rng_blocks, n_shift, n_psd)
    end

    synchronize()
    return nothing
end

# compute the generalized product :
# WᵀWx for symmetric cones 
# μH(s)x for symmetric cones

NVTX.@annotate "mul_Hs" function mul_Hs!(
    cones::CompositeConeGPU{T},
    y::CuVector{T},
    x::CuVector{T},
    work::CuVector{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_sparse_soc = cones.n_sparse_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim
    Hs = cones.Hs
    Hspsd = cones.Hspsd
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq
    w = cones.w
    η = cones.η

    mul_Hs_zero!(y, rng_cones, idx_eq)
    
    mul_Hs_nonnegative!(y, x, w, rng_cones, idx_inq)

    n_shift = n_linear
    if n_sparse_soc > 0 && n_sparse_soc <= SPARSE_SOC_PARALELL_NUM
        mul_Hs_soc_parallel_medium!(y, x, w, η, 
                            cones.worksoc1,
                            rng_cones, n_shift, n_sparse_soc)
        n_shift += n_sparse_soc
        n_soc -= n_sparse_soc
    end
    if n_soc > 0
        mul_Hs_soc!(y, x, w, η, rng_cones, n_shift, n_soc)
        n_shift += n_soc
    end

    n_nonsymmetric = n_exp + n_pow
    if n_nonsymmetric > 0
        mul_Hs_nonsymmetric!(y, Hs, x, rng_cones, n_shift, n_nonsymmetric)
        n_shift += n_nonsymmetric
    end

    if n_psd > 0
        mul_Hs_psd!(y, x, Hspsd, rng_cones, n_shift, n_psd, psd_dim)
    end

    synchronize()

    return nothing
end

NVTX.@annotate "mul_Hs_diag!" function mul_Hs_diag!(
    cones::CompositeConeGPU{T},
    y::CuVector{T},
    x::CuVector{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_sparse_soc = cones.n_sparse_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim
    Hs = cones.Hs
    Hspsd = cones.Hspsd
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq
    w = cones.w
    η = cones.η

    mul_Hs_zero!(y, rng_cones, idx_eq)
    
    mul_Hs_nonnegative!(y, x, w, rng_cones, idx_inq)

    if (n_soc - n_sparse_soc) > 0
        mul_Hs_dense_soc!(y, x, w, η, rng_cones, n_linear, n_sparse_soc, n_soc)
    end

    n_nonsymmetric = n_exp + n_pow
    if n_nonsymmetric > 0
        n_shift = n_linear+n_soc
        mul_Hs_nonsymmetric!(y, Hs, x, rng_cones, n_shift, n_nonsymmetric)
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        mul_Hs_psd!(y, x, Hspsd, rng_cones, n_shift, n_psd, psd_dim)
    end

    synchronize()
    return nothing
end

# x = λ ∘ λ for symmetric cone and x = s for asymmetric cones
NVTX.@annotate "affine_ds!" function affine_ds!(
    cones::CompositeConeGPU{T},
    ds::CuVector{T},
    s::CuVector{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_sparse_soc = cones.n_sparse_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq
    λ = cones.λ
    psd_dim = cones.psd_dim
    λpsd = cones.λpsd

    affine_ds_zero!(ds, rng_cones, idx_eq)

    affine_ds_nonnegative!(ds, λ, rng_cones, idx_inq)

    n_shift = n_linear
    if n_sparse_soc > 0 && n_sparse_soc <= SPARSE_SOC_PARALELL_NUM
        affine_ds_soc_parallel_medium!(ds, λ,
                            cones.worksoc1,
                            rng_cones, n_shift, n_sparse_soc)
        n_shift += n_sparse_soc
        n_soc -= n_sparse_soc
    end

    if n_soc > 0
        affine_ds_soc!(ds, λ, rng_cones, n_shift, n_soc)
        n_shift += n_soc
    end

    #update nonsymmetric cones
    n_nonsymmetric = n_exp + n_pow
    if n_nonsymmetric > 0
        affine_ds_nonsymmetric!(ds, s, rng_cones, n_shift, n_nonsymmetric)
        n_shift += n_nonsymmetric
    end

    if n_psd > 0
        affine_ds_psd_gpu!(ds,λpsd,rng_cones,psd_dim,n_shift,n_psd)
    end

    synchronize()
    return nothing
end

NVTX.@annotate "combined_ds_shift!" function combined_ds_shift!(
    cones::CompositeConeGPU{T},
    shift::CuVector{T},
    step_z::CuVector{T},
    step_s::CuVector{T},
    z::CuVector{T},
    σμ::T
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_sparse_soc = cones.n_sparse_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    grad = cones.grad
    H_dual = cones.H_dual
    αp = cones.αp
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq
    w = cones.w
    η = cones.η
    
    combined_ds_shift_zero!(shift, rng_cones, idx_eq)

    NVTX.NVTX.@range "Nonnegative" begin
        combined_ds_shift_nonnegative!(shift, step_z, step_s, σμ, rng_cones, idx_inq)
    end

    n_shift = n_linear
    NVTX.NVTX.@range "soc medium" begin
        if n_sparse_soc > 0 && n_sparse_soc <= SPARSE_SOC_PARALELL_NUM
            combined_ds_shift_soc_parallel_medium!(shift, step_z, step_s, w, η,
                                cones.worksoc1, cones.worksoc2, cones.worksoc3,
                                rng_cones, n_shift, n_sparse_soc, σμ)
            n_shift += n_sparse_soc
            n_soc -= n_sparse_soc
        end
    end
    
    if n_soc > 0
        combined_ds_shift_soc!(shift, step_z, step_s, w, η, rng_cones, n_shift, n_soc, σμ)
        n_shift += n_soc
    end

    if n_exp > 0
        combined_ds_shift_exp!(shift, step_z, step_s, z, grad, H_dual, rng_cones, σμ, n_shift, n_exp)
        n_shift += n_exp
    end

    if n_pow > 0
        combined_ds_shift_pow!(shift, step_z, step_s, z, grad, H_dual, αp, rng_cones, σμ, n_shift, n_exp, n_pow)
        n_shift += n_pow
    end
    
    if n_psd > 0 
        combined_ds_shift_psd!(cones,shift,step_z,step_s,n_shift,n_psd,σμ)
    end

    synchronize()
    return nothing

end

NVTX.@annotate "Δs_from_Δz_offset!" function Δs_from_Δz_offset!(
    cones::CompositeConeGPU{T},
    out::CuVector{T},
    ds::CuVector{T},
    work::CuVector{T},
    z::CuVector{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_sparse_soc = cones.n_sparse_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq
    w = cones.w
    λ = cones.λ
    η = cones.η

    Δs_from_Δz_offset_zero!(out, rng_cones, idx_eq)
    
    Δs_from_Δz_offset_nonnegative!(out, ds, z, rng_cones, idx_inq)

    n_shift = n_linear
    if n_sparse_soc > 0 && n_sparse_soc <= SPARSE_SOC_PARALELL_NUM
        Δs_from_Δz_offset_soc_parallel_medium!(out, ds, z, w, λ, η, 
                                                cones.worksoc1, cones.worksoc2, cones.worksoc3,
                                                rng_cones, n_shift, n_sparse_soc)
        n_shift += n_sparse_soc
        n_soc -= n_sparse_soc
    end

    if n_soc > 0
        Δs_from_Δz_offset_soc!(out, ds, z, w, λ, η, rng_cones, n_shift, n_soc)
        n_shift += n_soc
    end

    n_nonsymmetric = n_exp+n_pow
    if n_nonsymmetric > 0
        Δs_from_Δz_offset_nonsymmetric!(out, ds, rng_cones, n_shift, n_nonsymmetric)
        n_shift += n_nonsymmetric
    end

    if n_psd > 0
        Δs_from_Δz_offset_psd!(cones, out, ds, work, n_shift, n_psd)
    end

    synchronize()

    return nothing
end

# maximum allowed step length over all cones
NVTX.@annotate "step_length" function step_length(
     cones::CompositeConeGPU{T},
        dz::CuVector{T},
        ds::CuVector{T},
         z::CuVector{T},
         s::CuVector{T},
  settings::Settings{T},
      αmax::T,
) where {T}

    n_linear = cones.n_linear
    numel_linear = cones.numel_linear
    n_soc = cones.n_soc
    n_sparse_soc = cones.n_sparse_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    rng_cones       = cones.rng_cones
    idx_inq = cones.idx_inq
    α               = cones.α
    αp              = cones.αp
    Λisqrt = cones.Λisqrt
    eigvals = cones.eigvals
    d      = cones.workvec
    Rx     = cones.R
    Rinv   = cones.Rinv
    (workmat1, workmat2, workmat3) = (cones.workmat1, cones.workmat2, cones.workmat3)

    @. α = αmax          #Initialize step size

    step_length_nonnegative(dz, ds, z, s, α, αmax, rng_cones, idx_inq)
    @views αmax = min(αmax, minimum(α[1:numel_linear]))

    n_shift = n_linear
    n_parallel_soc = n_soc
    if n_sparse_soc > 0 && n_sparse_soc <= SPARSE_SOC_PARALELL_NUM
        αmax = step_length_soc_parallel_medium(dz, ds, z, s, 
                            cones.worksoc1, cones.worksoc2, cones.worksoc3,
                            αmax, rng_cones, n_shift, n_sparse_soc)
        n_shift += n_sparse_soc
        n_parallel_soc -= n_sparse_soc
    end

    if n_parallel_soc > 0
        αmax = step_length_soc(dz, ds, z, s, α, αmax, rng_cones, n_shift, n_parallel_soc)
        if αmax < 0
            throw(DomainError("starting point of line search not in SOC"))
        end
    end

    n_nonsymmetric = n_exp+n_pow
    if n_psd > 0
        n_shift = n_linear+n_soc+n_nonsymmetric
        αmax = step_length_psd(dz, ds, Λisqrt, eigvals, d, Rx, Rinv, workmat1, workmat2, workmat3, αmax, rng_cones, n_shift, n_psd)
        if αmax < 0
            throw(DomainError("starting point of line search not in positive semidefinite cones"))
        end
    end

    step = settings.linesearch_backtrack_step
    αmin = settings.min_terminate_step_length
    #if we have any nonsymmetric cones, then back off from full steps slightly
    #so that centrality checks and logarithms don't fail right at the boundaries
    if(n_nonsymmetric > 0)
        αmax = min(αmax,settings.max_step_fraction)
    end
  
    if n_exp > 0
        n_shift = n_linear+n_soc
        αmax = step_length_exp(dz, ds, z, s, α, rng_cones, αmax, αmin, step, n_shift, n_exp)
        if αmax < 0
            throw(DomainError("starting point of line search not in expotential cones"))
        end
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        αmax = step_length_pow(dz,ds,z,s,α,αp,rng_cones,αmax,αmin,step,n_shift,n_pow)
        if αmax < 0
            throw(DomainError("starting point of line search not in power cones"))
        end
    end

    return (αmax,αmax)
end

# compute the total barrier function at the point (z + α⋅dz, s + α⋅ds)
NVTX.@annotate "compute_barrier" function compute_barrier(
    cones::CompositeConeGPU{T},
    z::CuVector{T},
    s::CuVector{T},
    dz::CuVector{T},
    ds::CuVector{T},
    α::T
) where {T}

    n_linear = cones.n_linear
    n_nn = cones.n_nn
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim
    rng_cones = cones.rng_cones
    αp    = cones.αp
    workmat1 = cones.workmat1
    workvec  = cones.workvec

    barrier = zero(T)
    work = cones.α
    
    if n_nn > 0
        val = compute_barrier_nonnegative(work, z, s, dz, ds, α, rng_cones, cones.idx_inq, n_nn)
        barrier += val
    end

    if n_soc > 0
        val = compute_barrier_soc(work, z, s, dz, ds, α, rng_cones, n_linear, n_soc)
        barrier += val
    end

    if n_exp > 0
        n_shift = n_linear+n_soc
        val = compute_barrier_exp(work, z, s, dz, ds, α, rng_cones, n_shift, n_exp)
        barrier += val
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        val = compute_barrier_pow(work, z, s, dz, ds, α, αp, rng_cones, n_shift, n_pow)
        barrier += val
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        val = compute_barrier_psd(work, z, s, dz, ds, α, workmat1, workvec, rng_cones, psd_dim, n_shift, n_psd)
        barrier += val
    end

    return barrier
end

