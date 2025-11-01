
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
    
    (αmin,val) = margins_nonnegative(Val(n_nn > 0), z, α, rng_cones, idx_inq, αmin)
    β += val

    n_shift = n_linear
    (αmin,val) = margins_soc(Val(n_soc > 0), z, α, rng_cones, n_shift, n_soc, αmin)
    β += val

    n_shift = n_linear + n_soc
    (αmin,val) = margins_psd(Val(n_psd > 0), Z, z, eigvals, rng_cones, n_shift, n_psd, αmin)
    β += val

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

    scaled_unit_shift_soc!(Val(n_soc > 0), z, rng_cones, α, n_linear, n_soc)

    scaled_unit_shift_psd!(Val(n_psd > 0), z, α, rng_cones, psd_dim, n_linear + n_soc, n_psd)

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
    
    unit_initialization_soc!(Val(n_soc > 0), z, s, rng_cones, n_linear, n_soc)

    unit_initialization_exp!(Val(n_exp > 0), z, s, rng_cones, n_linear+n_soc, n_exp)

    unit_initialization_pow!(Val(n_pow > 0), z, s, αp, rng_cones, n_linear+n_soc+n_exp, n_pow)

    unit_initialization_psd!(Val(n_psd > 0), z, s, rng_cones, psd_dim, n_linear+n_soc+n_exp+n_pow, n_psd)

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

    set_identity_scaling_soc!(Val(n_soc > 0), w, cones.η, rng_cones, n_linear, n_soc)
    set_identity_scaling_soc_sparse!(Val(n_soc > 0), cones.d, cones.vut, rng_cones, n_linear, cones.n_sparse_soc)

    set_identity_scaling_psd!(Val(n_psd > 0), cones.R, cones.Rinv, cones.Hspsd, cones.psd_dim, n_psd)

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
    n_dense_soc = cones.n_dense_soc
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

    #Streams events
    streams = cones.streams
    events = cones.events

    update_scaling_nonnegative!(s, z, w, λ, rng_cones, idx_inq)

    update_scaling_soc!(sparse_soc_case(Val(n_sparse_soc)), s, z, w, λ, η, 
                                    cones.worksoc1, cones.worksoc2, cones.worksoc3, 
                                    rng_cones, n_linear, n_soc, n_dense_soc, n_sparse_soc, 
                                    streams[SOC_STREAM], events[SOC_STREAM])
    # off-diagonal update of SOCs
    update_scaling_soc_sparse!(sparse_soc_case(Val(n_sparse_soc)), w, η, d, vut, rng_cones, numel_linear, n_linear, n_sparse_soc, streams[SOC_STREAM], events[SOC_STREAM])

    update_scaling_exp!(Val(n_exp > 0), s, z, grad, Hs, H_dual, rng_cones, μ, scaling_strategy, n_linear+n_soc, n_exp, streams[EXP_STREAM], events[EXP_STREAM])

    update_scaling_pow!(Val(n_pow > 0), s, z, grad, Hs, H_dual, αp, rng_cones, μ, scaling_strategy, n_linear+n_soc+n_exp, n_exp, n_pow, streams[POW_STREAM], events[POW_STREAM])

    update_scaling_psd!(Val(n_psd > 0), cones.chol1, cones.chol2, cones.U, cones.S, cones.V, z, s, cones.workmat1, cones.λpsd, cones.Λisqrt, cones.R, cones.Rinv, cones.Hspsd, rng_cones, n_linear+n_soc+n_exp+n_pow, n_psd)

    cones_synchronize(n_soc, n_exp, n_pow, cones.events)

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

    #Streams events
    streams = cones.streams
    events = cones.events

    get_Hs_zero!(Hsblocks, rng_blocks, idx_eq)

    get_Hs_nonnegative!(Hsblocks, w, rng_cones, rng_blocks, idx_inq)

    get_Hs_soc!(sparse_soc_case(Val(n_sparse_soc)), Hsblocks, w, η, d, rng_cones, rng_blocks, n_linear, n_dense_soc, n_sparse_soc, streams[SOC_STREAM], events[SOC_STREAM])

    get_Hs_exp!(Val(n_exp > 0), Hsblocks, Hs, rng_blocks, n_linear+n_soc, n_exp, streams[EXP_STREAM], events[EXP_STREAM])

    get_Hs_pow!(Val(n_pow > 0), Hsblocks, Hs, rng_blocks, n_linear+n_soc+n_exp, n_exp, n_pow, streams[POW_STREAM], events[POW_STREAM])

    get_Hs_psd!(Val(n_psd > 0), Hsblocks, Hspsd, rng_blocks, n_linear+n_soc+n_exp+n_pow, n_psd)

    cones_synchronize(n_soc, n_exp, n_pow, cones.events)

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
    n_dense_soc = cones.n_dense_soc
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

    #Streams events
    streams = cones.streams
    events = cones.events

    mul_Hs_zero!(y, rng_cones, idx_eq)
    
    mul_Hs_nonnegative!(y, x, w, rng_cones, idx_inq)

    mul_Hs_soc!(sparse_soc_case(Val(n_sparse_soc)), y, x, w, η, cones.worksoc1, rng_cones, n_linear, n_soc, n_dense_soc, n_sparse_soc, streams[SOC_STREAM], events[SOC_STREAM])

    n_nonsymmetric = n_exp + n_pow
    mul_Hs_nonsymmetric!(Val(n_nonsymmetric > 0), y, Hs, x, rng_cones, n_linear+n_soc, n_nonsymmetric, streams[EXP_STREAM], events[EXP_STREAM])

    mul_Hs_psd!(Val(n_psd > 0), y, x, Hspsd, rng_cones, n_linear+n_soc+n_nonsymmetric, n_psd, psd_dim)

    cones_synchronize(n_soc, n_nonsymmetric, cones.events)

    return nothing
end

# NVTX.@annotate "mul_Hs_diag!" function mul_Hs_diag!(
#     cones::CompositeConeGPU{T},
#     y::CuVector{T},
#     x::CuVector{T}
# ) where {T}

#     n_linear = cones.n_linear
#     n_soc = cones.n_soc
#     n_sparse_soc = cones.n_sparse_soc
#     n_exp = cones.n_exp
#     n_pow = cones.n_pow
#     n_psd = cones.n_psd
#     psd_dim = cones.psd_dim
#     Hs = cones.Hs
#     Hspsd = cones.Hspsd
#     rng_cones = cones.rng_cones
#     idx_eq = cones.idx_eq
#     idx_inq = cones.idx_inq
#     w = cones.w
#     η = cones.η

#     mul_Hs_zero!(y, rng_cones, idx_eq)
    
#     mul_Hs_nonnegative!(y, x, w, rng_cones, idx_inq)

#     if (n_soc - n_sparse_soc) > 0
#         mul_Hs_dense_soc!(y, x, w, η, rng_cones, n_linear, n_soc, n_dense_soc, n_sparse_soc)
#     end

#     n_nonsymmetric = n_exp + n_pow
#     if n_nonsymmetric > 0
#         n_shift = n_linear+n_soc
#         mul_Hs_nonsymmetric!(y, Hs, x, rng_cones, n_shift, n_nonsymmetric)
#     end

#     if n_psd > 0
#         n_shift = n_linear+n_soc+n_exp+n_pow
#         mul_Hs_psd!(y, x, Hspsd, rng_cones, n_shift, n_psd, psd_dim)
#     end

#     synchronize()
#     return nothing
# end

# x = λ ∘ λ for symmetric cone and x = s for asymmetric cones
NVTX.@annotate "affine_ds!" function affine_ds!(
    cones::CompositeConeGPU{T},
    ds::CuVector{T},
    s::CuVector{T}
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_dense_soc = cones.n_dense_soc
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

    #Streams events
    streams = cones.streams
    events = cones.events

    affine_ds_zero!(ds, rng_cones, idx_eq)

    affine_ds_nonnegative!(ds, λ, rng_cones, idx_inq)

    affine_ds_soc!(sparse_soc_case(Val(n_sparse_soc)), ds, λ, cones.worksoc1, rng_cones, n_linear, n_soc, n_dense_soc, n_sparse_soc, streams[SOC_STREAM], events[SOC_STREAM])

    #update nonsymmetric cones
    n_nonsymmetric = n_exp + n_pow
    affine_ds_nonsymmetric!(Val(n_nonsymmetric > 0), ds, s, rng_cones, n_linear + cones.n_soc, n_nonsymmetric, streams[EXP_STREAM], events[EXP_STREAM])

    affine_ds_psd!(Val(n_psd > 0), ds, λpsd, rng_cones, psd_dim, n_linear+n_soc+n_nonsymmetric, n_psd)

    cones_synchronize(n_soc, n_nonsymmetric, cones.events)
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
    n_dense_soc = cones.n_dense_soc
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

    #Streams events
    streams = cones.streams
    events = cones.events

    combined_ds_shift_zero!(shift, rng_cones, idx_eq)

    # NVTX.@range "Nonnegative" begin
    combined_ds_shift_nonnegative!(shift, step_z, step_s, σμ, rng_cones, idx_inq)
    # end

    combined_ds_shift_soc!(sparse_soc_case(Val(n_sparse_soc)), shift, step_z, step_s, w, η,
                                cones.worksoc1, cones.worksoc2, cones.worksoc3,
                                rng_cones, n_linear, n_soc, n_dense_soc, n_sparse_soc, σμ, 
                                streams[SOC_STREAM], events[SOC_STREAM])

    combined_ds_shift_exp!(Val(n_exp > 0), shift, step_z, step_s, z, grad, H_dual, rng_cones, σμ, n_linear + n_soc, n_exp, streams[EXP_STREAM], events[EXP_STREAM])

    combined_ds_shift_pow!(Val(n_pow > 0), shift, step_z, step_s, z, grad, H_dual, αp, rng_cones, σμ, n_linear + n_soc + n_exp, n_exp, n_pow, streams[POW_STREAM], events[POW_STREAM])
    
    combined_ds_shift_psd!(Val(n_psd > 0), cones, shift, step_z, step_s, n_linear + n_soc + n_exp + n_pow, n_psd, σμ)

    cones_synchronize(n_soc, n_exp, n_pow, cones.events)

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
    n_dense_soc = cones.n_dense_soc
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

    #Streams events
    streams = cones.streams
    events = cones.events

    Δs_from_Δz_offset_zero!(out, rng_cones, idx_eq)
    
    Δs_from_Δz_offset_nonnegative!(out, ds, z, rng_cones, idx_inq)

    Δs_from_Δz_offset_soc!(sparse_soc_case(Val(n_sparse_soc)), out, ds, z, w, λ, η, 
                                                cones.worksoc1, cones.worksoc2, cones.worksoc3,
                                                rng_cones, n_linear, n_soc, n_dense_soc, n_sparse_soc,
                                                streams[SOC_STREAM], events[SOC_STREAM])

    n_nonsymmetric = n_exp+n_pow
    Δs_from_Δz_offset_nonsymmetric!(Val(n_nonsymmetric > 0), out, ds, rng_cones, n_linear + cones.n_soc, n_nonsymmetric, streams[EXP_STREAM], events[EXP_STREAM])
    
    Δs_from_Δz_offset_psd!(Val(n_psd > 0), cones, out, ds, work, n_linear + cones.n_soc + n_nonsymmetric, n_psd)

    cones_synchronize(n_soc, n_nonsymmetric, cones.events)

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
    n_dense_soc = cones.n_dense_soc
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

    #YC: to do multi streams
    @. α = αmax          #Initialize step size

    step_length_nonnegative(dz, ds, z, s, α, αmax, rng_cones, idx_inq)
    @views αmax = min(αmax, minimum(α[1:numel_linear]))

    αmax = step_length_soc(sparse_soc_case(Val(n_sparse_soc)), dz, ds, z, s,
                            cones.worksoc1, cones.worksoc2, cones.worksoc3, 
                            α, αmax, rng_cones,
                            n_linear, n_soc, n_dense_soc, n_sparse_soc)

    n_nonsymmetric = n_exp+n_pow
    αmax = step_length_psd(Val(n_psd > 0), dz, ds, Λisqrt, eigvals, d, Rx, Rinv, workmat1, workmat2, workmat3, αmax, rng_cones, n_linear+n_soc+n_nonsymmetric, n_psd)

    step = settings.linesearch_backtrack_step
    αmin = settings.min_terminate_step_length
    #if we have any nonsymmetric cones, then back off from full steps slightly
    #so that centrality checks and logarithms don't fail right at the boundaries
    αmax = fractional_step(Val(n_nonsymmetric > 0), αmax, settings.max_step_fraction)
  
    αmax = step_length_exp(Val(n_exp > 0), dz, ds, z, s, α, rng_cones, αmax, αmin, step, n_linear+n_soc, n_exp)

    αmax = step_length_pow(Val(n_pow > 0), dz, ds, z, s, α, αp, rng_cones, αmax, αmin, step, n_linear+n_soc+n_exp, n_pow)

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

    work = cones.α

    #YC: to do multi streams
    barrier_nn = compute_barrier_nonnegative(Val(n_nn > 0), work, z, s, dz, ds, α, rng_cones, cones.idx_inq, n_nn)

    barrier_soc = compute_barrier_soc(Val(n_soc > 0), work, z, s, dz, ds, α, rng_cones, n_linear, n_soc)

    barrier_exp = compute_barrier_exp(Val(n_exp > 0), work, z, s, dz, ds, α, rng_cones, n_linear+n_soc, n_exp)

    barrier_pow = compute_barrier_pow(Val(n_pow > 0), work, z, s, dz, ds, α, αp, rng_cones, n_linear+n_soc+n_exp, n_pow)

    barrier_psd = compute_barrier_psd(Val(n_psd > 0), work, z, s, dz, ds, α, workmat1, workvec, rng_cones, psd_dim, n_linear+n_soc+n_exp+n_pow, n_psd)

    return (barrier_nn + barrier_soc + barrier_exp + barrier_pow + barrier_psd)
end

####################################################
# Self-defined Synchronization
####################################################
# Synchronization for cones
@inline function cones_synchronize(
    n_soc::Cint,
    n_exp::Cint,
    n_pow::Cint,
    ev::Vector{CuEvent}
)
    cone_synchronize(Val(n_soc > 0), default_stream(), ev[SOC_STREAM])
    cone_synchronize(Val(n_exp > 0), default_stream(), ev[EXP_STREAM])
    cone_synchronize(Val(n_pow > 0), default_stream(), ev[POW_STREAM])
    synchronize()
end

@inline function cones_synchronize(
    n_soc::Cint,
    n_nonsymmetric::Cint,
    ev::Vector{CuEvent}
)
    cone_synchronize(Val(n_soc > 0), default_stream(), ev[SOC_STREAM])
    cone_synchronize(Val(n_nonsymmetric > 0), default_stream(), ev[EXP_STREAM])
    synchronize()
end

@inline function cone_synchronize(
    ::Val{false},
    st::CuStream,
    ev::CuEvent
)
    return nothing
end

@inline function cone_synchronize(
    ::Val{true},
    st::CuStream,
    ev::CuEvent
)
    CUDA.wait(ev, st)    
end

@inline function add_record(
    ::Val{false},
    st::CuStream,
    ev::CuEvent
)
    return nothing
end

@inline function add_record(
    ::Val{true},
    st::CuStream,
    ev::CuEvent
)
    record(ev, st)
end