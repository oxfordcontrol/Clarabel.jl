# ## ------------------------------------
# # Nonnegative Cone
# # -------------------------------------

# degree(K::NonnegativeCone{T}) where {T} = K.dim
# numel(K::NonnegativeCone{T}) where {T} = K.dim

# function rectify_equilibration!(
#     K::NonnegativeCone{T},
#     δ::AbstractVector{T},
#     e::AbstractVector{T}
# ) where{T}

#     #allow elementwise equilibration scaling
#     δ .= one(T)
#     return false
# end

function margins_nonnegative(
    ::Val{false},
    z::AbstractVector{T},
    α::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint},
    αmin::T
) where{T}
    return (αmin, zero(T))
end

function margins_nonnegative(
    ::Val{true},
    z::AbstractVector{T},
    α::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint},
    αmin::T
) where{T}
    margin = zero(T)
    
    CUDA.@allowscalar @inbounds for i in idx_inq
        rng_cone_i = rng_cones[i]
        @views zi = z[rng_cone_i]
        αmin = min(αmin,minimum(zi))
        @views αi = α[rng_cone_i]
        @. αi = max(zi,zero(T))
        synchronize()
        margin += sum(αi)
    end

    return (αmin, margin)
end

# place vector into nn cone
@inline function scaled_unit_shift_nonnegative!(
    z::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint},
    α::T
) where{T}

    CUDA.@allowscalar begin
        @inbounds for i in idx_inq
            rng_cone_i = rng_cones[i]
            @views @. z[rng_cone_i] += α 
        end
    end
end

# unit initialization for asymmetric solves
@inline function unit_initialization_nonnegative!(
   z::AbstractVector{T},
   s::AbstractVector{T},
   rng_cones::AbstractVector,
   idx_inq::Vector{Cint}
) where{T}

    CUDA.@allowscalar begin
        @inbounds for i in idx_inq
            rng_cone_i = rng_cones[i]
            @views @. z[rng_cone_i] = one(T)
            @views @. s[rng_cone_i] = one(T)
        end
    end
end

#configure cone internals to provide W = I scaling
@inline function set_identity_scaling_nonnegative!(
    w::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint}
) where {T}

    CUDA.@allowscalar begin
        @inbounds for i in idx_inq
            @views @. w[rng_cones[i]] = one(T)
        end
    end
end

@inline function update_scaling_nonnegative!(
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint}    
) where {T}

    CUDA.@allowscalar @inbounds for i in idx_inq
        rng_cone_i = rng_cones[i]
        @views  si = s[rng_cone_i]
        @views  zi = z[rng_cone_i]
        @views  @. λ[rng_cone_i] = sqrt(si*zi)
        @views  @. w[rng_cone_i] = sqrt(si/zi)
    end
end

@inline function get_Hs_nonnegative!(
    Hsblocks::AbstractVector{T},
    w::AbstractVector{T},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    idx_inq::Vector{Cint}  
) where {T}

    #this block is diagonal, and we expect here
    #to receive only the diagonal elements to fill
    CUDA.@allowscalar begin
        @inbounds for i in idx_inq
            @views wi = w[rng_cones[i]]
            @views @. Hsblocks[rng_blocks[i]] = wi^2
        end
    end
end

# compute the product y = WᵀWx
@inline function mul_Hs_nonnegative!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint} 
) where {T}

    #NB : seemingly sensitive to order of multiplication
    CUDA.@allowscalar begin
        @inbounds for i in idx_inq
            @views wi = w[rng_cones[i]]
            @views xi = x[rng_cones[i]]
            @views yi = y[rng_cones[i]]
            @. yi = (wi * (wi * xi))
        end
    end
end

# returns ds = λ∘λ for the nn cone
@inline function affine_ds_nonnegative!(
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint} 
) where {T}

    CUDA.@allowscalar begin
        @inbounds for i in idx_inq
            @views @. ds[rng_cones[i]] = λ[rng_cones[i]]^2
        end
    end
end

@inline function combined_ds_shift_nonnegative!(
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T,
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint}
) where {T}

    # The shift must be assembled carefully if we want to be economical with
    # allocated memory.  Will modify the step.z and step.s in place since
    # they are from the affine step and not needed anymore.
    #
    # We can't have aliasing vector arguments to gemv_W or gemv_Winv, so 
    # we need a temporary variable to assign #Δz <= WΔz and Δs <= W⁻¹Δs

    CUDA.@allowscalar begin
        @inbounds for i in idx_inq
            rng_i = rng_cones[i]
            @views shift_i = shift[rng_cones[i]]
            step_zi = step_z[rng_cones[i]]
            step_si = step_s[rng_cones[i]]

            #shift = W⁻¹Δs ∘ WΔz - σμe
            @. shift_i = step_si*step_zi - σμ    
        end
    end    
end

@inline function Δs_from_Δz_offset_nonnegative!(
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint} 
) where {T}
    CUDA.@allowscalar begin
        @inbounds for i in idx_inq
            @views @. out[rng_cones[i]] = ds[rng_cones[i]] / z[rng_cones[i]]
        end
    end
end

#return maximum allowable step length while remaining in the nn cone
function _kernel_step_length_nonnegative(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     len_rng::Cint,
     αmax::T
) where {T}


    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x
    if i <= len_rng
        αz = dz[i] < 0 ? (min(αmax,-z[i]/dz[i])) : αmax
        αs = ds[i] < 0 ? (min(αmax,-s[i]/ds[i])) : αmax
        α[i] = min(αz, αs)
    end

    return nothing
end

@inline function step_length_nonnegative(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     αmax::T,
     rng_cones::AbstractVector,
     idx_inq::Vector{Cint} 
) where {T}

    CUDA.@allowscalar begin
        @inbounds for i in idx_inq
            len_nn = Cint(length(rng_cones[i]))
            rng_cone_i = rng_cones[i]
            @views dzi = dz[rng_cone_i]
            @views dsi = ds[rng_cone_i]
            @views zi = z[rng_cone_i]
            @views si = s[rng_cone_i]
            @views αi = α[rng_cone_i]
            
            kernel = @cuda launch=false _kernel_step_length_nonnegative(dzi, dsi, zi, si, αi, len_nn, αmax)
            config = launch_configuration(kernel.fun)
            threads = min(len_nn, config.threads)
            blocks = cld(len_nn, threads)
        
            kernel(dzi, dsi, zi, si, αi, len_nn, αmax; threads, blocks)
        end
    end
    synchronize()
end

function _kernel_compute_barrier_nonnegative(
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    len_nn::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x
    if i <= len_nn
        barrier[i] = -logsafe((s[i] + α*ds[i])*(z[i] + α*dz[i]))
    end

    return nothing
end

@inline function compute_barrier_nonnegative(
    ::Val{false},
    work::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint},
    len_nn::Cint
) where {T}
    return zero(T)
end

@inline function compute_barrier_nonnegative(
    ::Val{true},
    work::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint},
    len_nn::Cint
) where {T}

    barrier = zero(T)
    CUDA.@allowscalar begin
        @inbounds for i in idx_inq
            rng_cone_i = rng_cones[i]
            @views worki = work[rng_cone_i]
            @views @. worki = -logsafe((s[rng_cone_i] + α*ds[rng_cone_i])*(z[rng_cone_i] + α*dz[rng_cone_i]))
            synchronize()
            barrier += sum(worki)
        end
    end

    return barrier
end

# # ---------------------------------------------
# # operations supported by symmetric cones only 
# # ---------------------------------------------

# # implements y = αWx + βy for the nn cone
# function mul_W!(
#     K::NonnegativeCone{T},
#     is_transpose::Symbol,
#     y::AbstractVector{T},
#     x::AbstractVector{T},
#     α::T,
#     β::T
# ) where {T}

#   #W is diagonal so ignore transposition
#   #@. y = α*(x*K.w) + β*y
#   @inbounds for i = eachindex(y)
#       y[i] = α*(x[i]*K.w[i]) + β*y[i]
#   end

#   return nothing
# end

# # implements y = αW^{-1}x + βy for the nn cone
# function mul_Winv!(
#     K::NonnegativeCone{T},
#     is_transpose::Symbol,
#     y::AbstractVector{T},
#     x::AbstractVector{T},
#     α::T,
#     β::T
# ) where {T}

#   #W is diagonal, so ignore transposition
#   #@. y = α*(x/K.w) + β.*y
#   @inbounds for i = eachindex(y)
#       y[i] = α*(x[i]/K.w[i]) + β*y[i]
#   end

#   return nothing
# end

# # implements x = λ \ z for the nn cone, where λ
# # is the internally maintained scaling variable.
# function λ_inv_circ_op!(
#     K::NonnegativeCone{T},
#     x::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     inv_circ_op!(K, x, K.λ, z)

# end

# # ---------------------------------------------
# # Jordan algebra operations for symmetric cones 
# # ---------------------------------------------

# # implements x = y ∘ z for the nn cone
# function circ_op!(
#     K::NonnegativeCone{T},
#     x::AbstractVector{T},
#     y::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     @. x = y*z

#     return nothing
# end

# # implements x = y \ z for the nn cone
# function inv_circ_op!(
#     K::NonnegativeCone{T},
#     x::AbstractVector{T},
#     y::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     @. x = z/y

#     return nothing
# end