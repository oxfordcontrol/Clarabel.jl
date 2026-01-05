# # ----------------------------------------------------
# # Second Order Cone
# # ----------------------------------------------------

function _kernel_margins_soc(
    z::AbstractVector{T},
    α::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where{T}
    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        
        val = zero(T)
        @inbounds for j in (rng_cone_i.start + one(Cint)):rng_cone_i.stop 
            val += z[j]*z[j]
        end
        α[i]  = z[rng_cone_i.start] - sqrt(val)
    end

    return nothing
end

function margins_soc(
    ::Val{false},
    z::AbstractVector{T},
    α::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint,
    αmin::T
) where{T}
    return (αmin, zero(T))
end

function margins_soc(
    ::Val{true},
    z::AbstractVector{T},
    α::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint,
    αmin::T
) where{T}
    kernel = @cuda launch=false _kernel_margins_soc(z, α, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(z, α, rng_cones, n_shift, n_soc; threads, blocks)

    @views αsoc = α[1:n_soc]
    αmin = min(αmin,minimum(αsoc))
    CUDA.@sync @. αsoc = max(zero(T),αsoc)
    return (αmin, sum(αsoc))
end

# place vector into socone
function _kernel_scaled_unit_shift_soc!(
    z::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where{T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        z[rng_cone_i.start] += α
    end

    return nothing
end

function scaled_unit_shift_soc!(
    ::Val{false},
    z::AbstractVector{T},
    rng_cones::AbstractVector,
    α::T,
    n_shift::Cint,
    n_soc::Cint   
) where{T}
    return nothing
end

function scaled_unit_shift_soc!(
    ::Val{true},
    z::AbstractVector{T},
    rng_cones::AbstractVector,
    α::T,
    n_shift::Cint,
    n_soc::Cint   
) where{T}

    kernel = @cuda launch=false _kernel_scaled_unit_shift_soc!(z, α, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    kernel(z, α, rng_cones, n_shift, n_soc; threads, blocks)
end

# unit initialization for asymmetric solves
function _kernel_unit_initialization_soc!(
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where{T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        z[rng_cone_i.start] = one(T)
        @inbounds for j in (rng_cone_i.start + one(Cint)):rng_cone_i.stop 
            z[j] = zero(T)
        end

        s[rng_cone_i.start] = one(T)
        @inbounds for j in (rng_cone_i.start + one(Cint)):rng_cone_i.stop 
            s[j] = zero(T)
        end
    end
 
    return nothing
end 

@inline function unit_initialization_soc!(
    ::Val{false},
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where{T}
    return nothing
end 

@inline function unit_initialization_soc!(
    ::Val{true},
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where{T}

    kernel = @cuda launch=false _kernel_unit_initialization_soc!(z, s, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    kernel(z, s, rng_cones, n_shift, n_soc; threads, blocks)
end 

# # configure cone internals to provide W = I scaling
function _kernel_set_identity_scaling_soc!(
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_soc
        rng_cone_i = rng_cones[i + n_linear]

        w[rng_cone_i.start] = one(T)
        @inbounds for j in (rng_cone_i.start + one(Cint)):rng_cone_i.stop 
            w[j] = zero(T)
        end
        η[i]  = one(T)
    end

    return nothing
end

@inline function set_identity_scaling_soc!(
    ::Val{false},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}
    return nothing
end

@inline function set_identity_scaling_soc!(
    ::Val{true},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}
    kernel = @cuda launch=false _kernel_set_identity_scaling_soc!(w, η, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    kernel(w, η, rng_cones, n_shift, n_soc; threads, blocks)
end

@inline function set_identity_scaling_soc_sparse!(
    ::Val{false},
    d::AbstractVector{T},
    vut::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}
    return nothing
end

@inline function set_identity_scaling_soc_sparse!(
    ::Val{true},
    d::AbstractVector{T},
    vut::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}
    fill!(vut, 0)

    shift = 1
    CUDA.@allowscalar for i in 1:n_sparse_soc
        d[i]  = T(0.5)
        len_i = length(rng_cones[i + n_shift])
        vut[shift+len_i] = sqrt(T(0.5))
        shift += 2*len_i
    end
end

function _kernel_update_scaling_soc_dense!(
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_sparse_soc::Cint,
    n_dense_soc::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_dense_soc
        rng_i = rng_cones[i + n_linear + n_sparse_soc]
        @views zi = z[rng_i] 
        @views si = s[rng_i] 
        @views wi = w[rng_i] 
        @views λi = λ[rng_i]

        #first calculate the scaled vector w
        @views zscale = _sqrt_soc_residual_gpu(zi)
        @views sscale = _sqrt_soc_residual_gpu(si)

        #the leading scalar term for W^TW
        η[i + n_sparse_soc] = sqrt(sscale/zscale)

        # construct w and normalize
        @inbounds for k in rng_i
            w[k] = s[k]/(sscale)
        end

        wi[1]  += zi[1]/(zscale)

        @inbounds for j in 2:length(wi)
            wi[j] -= zi[j]/(zscale)
        end
    
        wscale = _sqrt_soc_residual_gpu(wi)
        @inbounds for j in eachindex(wi)
            wi[j] /= wscale
        end

        #try to force badly scaled w to come out normalized
        w1sq = zero(T)
        @inbounds for j in 2:length(wi)
            w1sq += wi[j]*wi[j]
        end
        wi[1] = sqrt(1 + w1sq)

        #Compute the scaling point λ.   Should satisfy λ = Wz = W^{-T}s
        γi = 0.5 * wscale
        λi[1] = γi 

        coef = inv(si[1]/sscale + zi[1]/zscale + 2*γi)
        coef2 = sqrt(sscale*zscale)
        coef *= coef2
        c1 = ((γi + zi[1]/zscale)/sscale)
        c2 = ((γi + si[1]/sscale)/zscale)
        @inbounds for j in 2:length(λi)
            λi[j] = coef*(c1*si[j] +c2*zi[j])
        end
        λi[1] *= coef2 
    end

    return nothing
end

#Update dense block for SOC
@inline function update_scaling_soc_dense!(
    ::Val{false},
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_sparse_soc::Cint,
    n_dense_soc::Cint,
    st::CuStream
) where {T}
    return nothing
end

@inline function update_scaling_soc_dense!(
    ::Val{true},
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_sparse_soc::Cint,
    n_dense_soc::Cint,
    st::CuStream
) where {T}
    kernel = @cuda launch=false _kernel_update_scaling_soc_dense!(s, z, w, λ, η, rng_cones, n_linear, n_sparse_soc, n_dense_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_dense_soc, config.threads)
    blocks = cld(n_dense_soc, threads)

    kernel(s, z, w, λ, η, rng_cones, n_linear, n_sparse_soc, n_dense_soc; threads, blocks, stream = st)
end

#Case 0: n_sparse_soc = 0
@inline function update_scaling_soc!(
    ::Val{0},
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    workz::Nothing,
    works::Nothing,
    workw::Nothing,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    update_scaling_soc_dense!(Val(n_soc > 0), s, z, w, λ, η, rng_cones, n_linear, n_sparse_soc, n_dense_soc, st)
    add_record(Val(n_soc > 0), st, ev)
end

#Case 2: n_sparse_soc > SPARSE_SOC_PARALELL_NUM
@inline function update_scaling_soc!(
    ::Val{2},
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    workz::AbstractVector{T},
    works::AbstractVector{T},
    workw::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    update_scaling_soc_dense!(Val(n_soc > 0), s, z, w, λ, η, rng_cones, n_linear, n_sparse_soc, n_dense_soc, st)
end


@inline function _update_scaling_soc_sparse!(
    w::AbstractVector{T},
    u::AbstractVector{T},
    v::AbstractVector{T},
    η::T
) where {T}

    #Populate sparse expansion terms if allocated
    #various intermediate calcs for u,v,d
    α  = 2*w[1]

    #Scalar d is the upper LH corner of the diagonal
    #term in the rank-2 update form of W^TW
    
    wsq    = 2*w[1]*w[1] - 1
    wsqinv = 1/wsq
    d    = wsqinv / 2

    #the vectors for the rank two update
    #representation of W^TW
    u0  = sqrt(wsq - d)
    u1 = α/u0
    v0 = zero(T)
    v1 = sqrt( 2*(2 + wsqinv)/(2*wsq - wsqinv))
    
    minus_η2 = -η*η 
    u[1] = minus_η2*u0
    @views u[2:end] .= minus_η2.*u1.*w[2:end]
    v[1] = minus_η2*v0
    @views v[2:end] .= minus_η2.*v1.*w[2:end]
    synchronize()

    return d
end

@inline function _kernel_update_scaling_soc_sparse!(
    w::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    vut::AbstractVector{T},
    rng_cones::AbstractVector,
    numel_linear::Cint,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}
    
    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_sparse_soc 
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        len_i = length(rng_i)
        rng_sparse_i = rng_i .- numel_linear
        startidx = 2*(rng_sparse_i.stop - len_i)
        wi = view(w, rng_i)
        vi = view(vut, (startidx+1):(startidx+len_i))
        ui = view(vut, (startidx+len_i+1):(startidx+2*len_i))

        #Unroll function _update_scaling_soc_sparse!()
        #Populate sparse expansion terms if allocated
        #various intermediate calcs for u,v,d
        α  = 2*wi[1]

        #Scalar d is the upper LH corner of the diagonal
        #term in the rank-2 update form of W^TW
        
        wsq    = 2*wi[1]*wi[1] - 1
        wsqinv = 1/wsq
        di    = wsqinv / 2
        d[i]   = di

        #the vectors for the rank two update
        #representation of W^TW
        u0  = sqrt(wsq - di)
        u1 = α/u0
        v0 = zero(T)
        v1 = sqrt( 2*(2 + wsqinv)/(2*wsq - wsqinv))
        
        minus_η2 = -η[i]*η[i]
        ui[1] = minus_η2*u0
        vi[1] = minus_η2*v0

        @inbounds for j in 2:length(ui)
            ui[j] = minus_η2*u1*wi[j]
            vi[j] = minus_η2*v1*wi[j]
        end
    end
end

@generated function sparse_soc_case(::Val{N}) where {N}
    if N == 0
        return :(Val(0))  # No SOCs
    elseif N <= SPARSE_SOC_PARALELL_NUM
        return :(Val(1))  # Medium
    else
        return :(Val(2))  # Large
    end
end

@inline function update_scaling_soc_sparse!(
    ::Val{0},
    w::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    vut::AbstractVector{T},
    rng_cones::AbstractVector,
    numel_linear::Cint,
    n_shift::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    return nothing
end

@inline function update_scaling_soc_sparse!(
    ::Val{2},
    w::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    vut::AbstractVector{T},
    rng_cones::AbstractVector,
    numel_linear::Cint,
    n_shift::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}

    kernel = @cuda launch=false _kernel_update_scaling_soc_sparse!(w, η, d, vut, rng_cones, numel_linear, n_shift, n_sparse_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_sparse_soc, config.threads)
    blocks = cld(n_sparse_soc, threads)

    kernel(w, η, d, vut, rng_cones, numel_linear, n_shift, n_sparse_soc; threads, blocks, stream=st)
    record(ev, st)
end

function _kernel_get_Hs_soc_dense!(
    Hsblocks::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_dense_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        rng_block_i = rng_blocks[shift_i]
        size_i = length(rng_cone_i)
        @views wi = w[rng_cone_i] 
        @views Hsblocki = Hsblocks[rng_block_i]

        hidx = one(Cint)
        @inbounds for col in rng_cone_i
            wcol = w[col]
            @inbounds for row in rng_cone_i
                Hsblocki[hidx] = 2*w[row]*wcol
                hidx += 1
            end 
        end
        Hsblocki[1] -= one(T)
        @inbounds for ind in 2:size_i
            Hsblocki[(ind-1)*size_i + ind] += one(T)
        end
        η2 = η[n_sparse_soc+i]^2
        @inbounds for j in eachindex(Hsblocki)
            Hsblocki[j] *= η2 
        end

    end

    return nothing
end

@inline function get_Hs_soc_dense!(
    ::Val{false},
    Hsblocks::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream
) where {T}
    return nothing
end

@inline function get_Hs_soc_dense!(
    ::Val{true},
    Hsblocks::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream
) where {T}

    kernel = @cuda launch=false _kernel_get_Hs_soc_dense!(Hsblocks, w, η, rng_cones, rng_blocks, n_shift, n_dense_soc, n_sparse_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_dense_soc, config.threads)
    blocks = cld(n_dense_soc, threads)

    kernel(Hsblocks, w, η, rng_cones, rng_blocks, n_shift, n_dense_soc, n_sparse_soc; threads, blocks, stream=st)

end

function _kernel_get_Hs_soc_sparse_parallel!(
    Hsblocks::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_sparse_soc
        shift_i = i + n_shift
        rng_block_i = rng_blocks[shift_i]

        η2 = η[i]^2
        @inbounds for col in rng_block_i
            Hsblocks[col] = η2
        end
        Hsblocks[rng_block_i.start] *= d[i]

    end

    return nothing
end

@inline function get_Hs_soc!(
    ::Val{0},
    Hsblocks::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    get_Hs_soc_dense!(Val(n_dense_soc > 0), Hsblocks, w, η, rng_cones, rng_blocks, n_shift, n_dense_soc, n_sparse_soc, st)
    add_record(Val(n_dense_soc > 0), st, ev)
end

@inline function get_Hs_soc!(
    ::Val{2},
    Hsblocks::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}

    #For sparse form, we are returning here the diagonal D block 
    #from the sparse representation of W^TW, but not the
    #extra two entries at the bottom right of the block.
    #The AbstractVector for s and z (and its views) don't
    #know anything about the 2 extra sparsifying entries
    kernel = @cuda launch=false _kernel_get_Hs_soc_sparse_parallel!(Hsblocks, η, d, rng_blocks, n_shift, n_sparse_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_sparse_soc, config.threads)
    blocks = cld(n_sparse_soc, threads)

    kernel(Hsblocks, η, d, rng_blocks, n_shift, n_sparse_soc; threads, blocks, stream=st)

    #update dense socs
    get_Hs_soc_dense!(Val(n_dense_soc > 0), Hsblocks, w, η, rng_cones, rng_blocks, n_shift + n_sparse_soc, n_dense_soc, n_sparse_soc, st)

    record(ev, st)
end

# compute the product y = WᵀWx
function _kernel_mul_Hs_soc!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    # y = = H^{-1}x = W^TWx
    # where H^{-1} = \eta^{2} (2*ww^T - J)
    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views xi = x[rng_cone_i] 
        @views yi = y[rng_cone_i] 
        @views wi = w[rng_cone_i] 

        c = 2*_dot_xy_gpu(wi,xi,1:size_i)

        yi[1] = -xi[1] + c*wi[1]
        @inbounds for j in 2:size_i
            yi[j] = xi[j] + c*wi[j]
        end

        _multiply_gpu(yi,η[i]^2)
    end

    return nothing
end

@inline function mul_Hs_soc!(
    ::Val{0},
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    work::Nothing,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    mul_Hs_dense_soc!(Val(n_dense_soc > 0), y, x, w, η, rng_cones, n_linear, n_soc, st)
    add_record(Val(n_soc > 0), st, ev)
end

@inline function mul_Hs_soc!(
    ::Val{2},
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    work::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}

    #multiply dense socs
    kernel = @cuda launch=false _kernel_mul_Hs_soc!(y, x, w, η, rng_cones, n_linear, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    kernel(y, x, w, η, rng_cones, n_linear, n_soc; threads, blocks, stream=st)
    record(ev, st)
end

#multiplication on dense socs
@inline function mul_Hs_dense_soc!(
    ::Val{false},
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    st::CuStream
) where {T}
    return nothing
end

@inline function mul_Hs_dense_soc!(
    ::Val{true},
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    st::CuStream
) where {T}

    kernel = @cuda launch=false _kernel_mul_Hs_soc!(y, x, w, η, rng_cones, n_shift, n_dense_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_dense_soc, config.threads)
    blocks = cld(n_dense_soc, threads)

    kernel(y, x, w, η, rng_cones, n_shift, n_dense_soc; threads, blocks, stream=st)
end

# returns x = λ ∘ λ for the socone
function _kernel_affine_ds_soc!(
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        @views dsi = ds[rng_cone_i] 
        @views λi = λ[rng_cone_i] 

        #circ product λ∘λ
        dsi[1] = zero(T)
        for j in 1:length(dsi)
            dsi[1] += λi[j]*λi[j]
        end
        λi0 = λi[1]
        for j = 2:length(dsi)
            dsi[j] = 2*λi0*λi[j]
        end
      
    end

    return nothing

end

@inline function affine_ds_soc!(
    ::Val{0},
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    work::Nothing,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    affine_ds_dense_soc!(Val(n_soc > 0), ds, λ, rng_cones, n_linear, n_soc, st)
    add_record(Val(n_soc > 0), st, ev)
end

@inline function affine_ds_soc!(
    ::Val{2},
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    work::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    affine_ds_dense_soc!(Val(n_soc > 0), ds, λ, rng_cones, n_linear, n_soc, st)
    record(ev, st)
end

@inline function affine_ds_dense_soc!(
    ::Val{false},
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    st::CuStream
) where {T}
    return nothing
end

@inline function affine_ds_dense_soc!(
    ::Val{true},
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    st::CuStream
) where {T}

    kernel = @cuda launch=false _kernel_affine_ds_soc!(ds, λ, rng_cones, n_shift, n_dense_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_dense_soc, config.threads)
    blocks = cld(n_dense_soc, threads)

    kernel(ds, λ, rng_cones, n_shift, n_dense_soc; threads, blocks, stream=st)
end

function _kernel_combined_ds_shift_soc!(
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint,
    σμ::T
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views step_zi = step_z[rng_cone_i] 
        @views step_si = step_s[rng_cone_i] 
        @views wi = w[rng_cone_i] 
        @views shifti = shift[rng_cone_i] 
    
        #shift vector used as workspace for a few steps 
        tmp = shifti            

        #Δz <- WΔz
        @inbounds for j in 1:size_i
            tmp[j] = step_zi[j]
        end         
        ζ = zero(T)
        
        @inbounds for j in 2:size_i
            ζ += wi[j]*tmp[j]
        end

        c = tmp[1] + ζ/(1+wi[1])
      
        step_zi[1] = η[i]*(wi[1]*tmp[1] + ζ)
      
        @inbounds for j in 2:size_i
            step_zi[j] = η[i]*(tmp[j] + c*wi[j]) 
        end      

        #Δs <- W⁻¹Δs
        @inbounds for j in 1:size_i
            tmp[j] = step_si[j]
        end           
        ζ = zero(T)
        @inbounds for j in 2:size_i
            ζ += wi[j]*tmp[j]
        end

        c = -tmp[1] + ζ/(1+wi[1])
    
        step_si[1] = (one(T)/η[i])*(wi[1]*tmp[1] - ζ)
    
        @inbounds for j = 2:size_i
            step_si[j] = (one(T)/η[i])*(tmp[j] + c*wi[j])
        end

        #shift = W⁻¹Δs ∘ WΔz - σμe  
        val = zero(T)
        @inbounds for j in 1:size_i
            val += step_si[j]*step_zi[j]
        end       
        shifti[1] = val - σμ 

        s0   = step_si[1]
        z0   = step_zi[1]
        for j = 2:size_i
            shifti[j] = s0*step_zi[j] + z0*step_si[j]
        end      
    end                    

    return nothing
end

@inline function combined_ds_shift_soc!(
    ::Val{0},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    workz::Nothing,
    works::Nothing,
    work::Nothing,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    σμ::T,
    st::CuStream,
    ev::CuEvent
) where {T}
    combined_ds_shift_dense_soc!(Val(n_soc > 0), shift, step_z, step_s, w, η, rng_cones, n_linear, n_soc, σμ, st)
    add_record(Val(n_soc > 0), st, ev)
end

@inline function combined_ds_shift_soc!(
    ::Val{2},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    workz::AbstractVector{T}, 
    works::AbstractVector{T}, 
    work::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    σμ::T,
    st::CuStream,
    ev::CuEvent
) where {T}
    combined_ds_shift_dense_soc!(Val(n_soc > 0), shift, step_z, step_s, w, η, rng_cones, n_linear, n_soc, σμ, st)
    record(ev, st)
end

@inline function combined_ds_shift_dense_soc!(
    ::Val{false},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    σμ::T,
    st::CuStream
) where {T}
    return nothing
end

@inline function combined_ds_shift_dense_soc!(
    ::Val{true},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    σμ::T,
    st::CuStream
) where {T}

    kernel = @cuda launch=false _kernel_combined_ds_shift_soc!(shift, step_z, step_s, w, η, rng_cones, n_shift, n_dense_soc, σμ)
    config = launch_configuration(kernel.fun)
    threads = min(n_dense_soc, config.threads)
    blocks = cld(n_dense_soc, threads)

    kernel(shift, step_z, step_s, w, η, rng_cones, n_shift, n_dense_soc, σμ; threads, blocks, stream=st)
end

function _kernel_Δs_from_Δz_offset_soc!(
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views outi = out[rng_cone_i] 
        @views dsi = ds[rng_cone_i] 
        @views zi = z[rng_cone_i] 
        @views wi = w[rng_cone_i] 
        @views λi = λ[rng_cone_i] 

        #out = Wᵀ(λ \ ds).  Below is equivalent,
        #but appears to be a little more stable 
        reszi = _soc_residual_gpu(zi)

        @views λ1ds1  = _dot_xy_gpu(λi,dsi,2:size_i)
        @views w1ds1  = _dot_xy_gpu(wi,dsi,2:size_i)

        _minus_vec_gpu(outi,zi)
        outi[1] = zi[1]
    
        c = λi[1]*dsi[1] - λ1ds1
        _multiply_gpu(outi,c/reszi)

        outi[1] += η[i]*w1ds1
        @inbounds for j in 2:size_i
            outi[j] += η[i]*(dsi[j] + w1ds1/(1+wi[1])*wi[j])
        end
    
        _multiply_gpu(outi,one(T)/λi[1])
    end

    return nothing

end

@inline function Δs_from_Δz_offset_soc!(
    ::Val{0},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    workz::Nothing,
    workλ::Nothing,
    workw::Nothing,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    Δs_from_Δz_offset_dense_soc!(Val(n_soc > 0), out, ds, z, w, λ, η, rng_cones, n_linear, n_soc, st)
    add_record(Val(n_soc > 0), st, ev)
end

@inline function Δs_from_Δz_offset_soc!(
    ::Val{2},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    workz::AbstractVector{T}, 
    workλ::AbstractVector{T}, 
    workw::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    Δs_from_Δz_offset_dense_soc!(Val(n_soc > 0), out, ds, z, w, λ, η, rng_cones, n_linear, n_soc, st)
    record(ev, st)
end

@inline function Δs_from_Δz_offset_dense_soc!(
    ::Val{false},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    st::CuStream
) where {T}
    return nothing
end

@inline function Δs_from_Δz_offset_dense_soc!(
    ::Val{true},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    st::CuStream
) where {T}

    kernel = @cuda launch=false _kernel_Δs_from_Δz_offset_soc!(out, ds, z, w, λ, η, rng_cones, n_shift, n_dense_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_dense_soc, config.threads)
    blocks = cld(n_dense_soc, threads)

    kernel(out, ds, z, w, λ, η, rng_cones, n_shift, n_dense_soc; threads, blocks, stream=st)
end

#return maximum allowable step length while remaining in the socone
function _kernel_step_length_soc(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     rng_cones::AbstractVector,
     n_shift::Cint,
     n_soc::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        @views si = s[rng_cone_i] 
        @views dsi = ds[rng_cone_i] 
        @views zi = z[rng_cone_i] 
        @views dzi = dz[rng_cone_i]         

        αz   = _step_length_soc_component_gpu(zi,dzi,α[i])
        αs   = _step_length_soc_component_gpu(si,dsi,α[i])
        α[i] = min(αz,αs)
    end

    return nothing
end

@inline function step_length_soc(
    ::Val{0},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    worka::Nothing,
    workb::Nothing,
    workc::Nothing,
    α::AbstractVector{T},
    αmax::T,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint
) where {T}
    αmax = step_length_dense_soc(Val(n_soc > 0), dz, ds, z, s, α, αmax, rng_cones, n_linear, n_soc)
    return αmax
end

@inline function step_length_soc(
    ::Val{2},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    worka::AbstractVector{T},
    workb::AbstractVector{T},
    workc::AbstractVector{T},
    α::AbstractVector{T},
    αmax::T,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint
) where {T}
    αmax = step_length_dense_soc(Val(n_soc > 0), dz, ds, z, s, α, αmax, rng_cones, n_linear, n_soc)
    return αmax
end

@inline function step_length_dense_soc(
    ::Val{false},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     αmax::T,
     rng_cones::AbstractVector,
     n_shift::Cint,
     n_dense_soc::Cint
) where {T}
    return αmax
end

@inline function step_length_dense_soc(
    ::Val{true},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     αmax::T,
     rng_cones::AbstractVector,
     n_shift::Cint,
     n_dense_soc::Cint
) where {T}

    kernel = @cuda launch=false _kernel_step_length_soc(dz, ds, z, s, α, rng_cones, n_shift, n_dense_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_dense_soc, config.threads)
    blocks = cld(n_dense_soc, threads)

    CUDA.@sync kernel(dz, ds, z, s, α, rng_cones, n_shift, n_dense_soc; threads, blocks)
    @views αmax = min(αmax,minimum(α[1:n_dense_soc]))

    if αmax < 0
            throw(DomainError("starting point of line search not in SOC"))
    end

    return αmax
end

function _kernel_compute_barrier_soc(
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        @views si = s[rng_cone_i] 
        @views dsi = ds[rng_cone_i] 
        @views zi = z[rng_cone_i] 
        @views dzi = dz[rng_cone_i]  
        res_si = _soc_residual_shifted(si,dsi,α)
        res_zi = _soc_residual_shifted(zi,dzi,α)

        # avoid numerical issue if res_s <= 0 or res_z <= 0
        barrier[i] = (res_si > 0 && res_zi > 0) ? -logsafe(res_si*res_zi)/2 : Inf
    end

    return nothing
end

@inline function compute_barrier_soc(
    ::Val{false},
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}
    return zero(T)
end

@inline function compute_barrier_soc(
    ::Val{true},
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    kernel = @cuda launch=false _kernel_compute_barrier_soc(barrier,z,s,dz,ds,α,rng_cones,n_linear,n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(barrier,z,s,dz,ds,α,rng_cones,n_linear,n_soc; threads, blocks)

    return sum(barrier[1:n_soc])
end

# ---------------------------------------------
# internal operations for second order cones 
# ---------------------------------------------

@inline function _soc_residual_gpu(z::AbstractVector{T}) where {T} 
    res = z[1]*z[1]
    @inbounds for j in 2:length(z)
        res -= z[j]*z[j]
    end
    
    return res
end 

@inline function _sqrt_soc_residual_gpu(z::AbstractVector{T}) where {T} 
    res = _soc_residual_gpu(z)
    
    # set res to 0 when z is not an interior point
    res = res > 0.0 ? sqrt(res) : zero(T)
end 

@inline function _dot_xy_gpu(x::AbstractVector{T},y::AbstractVector{T},rng::UnitRange) where {T} 
    val = zero(T)
    @inbounds for j in rng
        val += x[j]*y[j]
    end
    
    return val
end 

@inline function _minus_vec_gpu(y::AbstractVector{T},x::AbstractVector{T}) where {T} 
    @inbounds for j in 1:length(x)
        y[j] = -x[j]
    end
end 

@inline function _multiply_gpu(x::AbstractVector{T},a::T) where {T} 
    @inbounds for j in 1:length(x)
        x[j] *= a 
    end
end 

# find the maximum step length α≥0 so that
# x + αy stays in the SOC
@inline function _step_length_soc_component_gpu(
    x::AbstractVector{T},
    y::AbstractVector{T},
    αmax::T
) where {T}

    if x[1] >= 0 && y[1] < 0
        αmax = min(αmax,-x[1]/y[1])
    end

    # assume that x is in the SOC, and find the minimum positive root
    # of the quadratic equation:  ||x₁+αy₁||^2 = (x₀ + αy₀)^2

    @views a = _soc_residual_gpu(y) #NB: could be negative
    @views b = 2*(x[1]*y[1] - _dot_xy_gpu(x,y,2:length(x)))
    @views c = max(zero(T),_soc_residual_gpu(x)) #should be ≥0
    d = b^2 - 4*a*c

    if(c < 0)
        # This should never be reachable since c ≥ 0 above
        return -Inf
    end

    if( (a > 0 && b > 0) || d < 0)
        #all negative roots / complex root pair
        #-> infinite step length
        return αmax

    elseif a == 0
        #edge case with only one root.  This corresponds to
        #the case where the search direction is exactly on the 
        #cone boundary.   The root should be -c/b, but b can't 
        #be negative since both (x,y) are in the cone and it is 
        #self dual, so <x,y> \ge 0 necessarily.
        return αmax

    elseif c == 0
        #Edge case with one of the roots at 0.   This corresponds 
        #to the case where the initial point is exactly on the 
        #cone boundary.  The other root is -b/a.   If the search 
        #direction is in the cone, then a >= 0 and b can't be 
        #negative due to self-duality.  If a < 0, then the 
        #direction is outside the cone and b can't be positive.
        #Either way, step length is determined by whether or not 
        #the search direction is in the cone.

        return (a >= 0 ? αmax : zero(T)) 
    end 


    # if we got this far then we need to calculate a pair 
    # of real roots and choose the smallest positive one.  
    # We need to be cautious about cancellations though.  
    # See §1.4: Goldberg, ACM Computing Surveys, 1991 
    # https://dl.acm.org/doi/pdf/10.1145/103162.103163

    t = (b >= 0) ? (-b - sqrt(d)) : (-b + sqrt(d))

    r1 = (2*c)/t;
    r2 = t/(2*a);

    #return the minimum positive root, up to αmax
    r1 = r1 < 0 ? floatmax(T) : r1
    r2 = r2 < 0 ? floatmax(T) : r2

    return min(αmax,r1,r2)

end

# parallel implementation for the step size computation for it
@inline function _step_length_soc_component_gpu_2(
    x::AbstractVector{T},
    y::AbstractVector{T},
    αmax::T
) where {T}

    if x[1] >= 0 && y[1] < 0
        αmax = min(αmax,-x[1]/y[1])
    end

    # assume that x is in the SOC, and find the minimum positive root
    # of the quadratic equation:  ||x₁+αy₁||^2 = (x₀ + αy₀)^2

    @views a = y[1]*y[1] - dot(y[2:end], y[2:end]) #NB: could be negative
    @views b = 2*(x[1]*y[1] - dot(x[2:end],y[2:end]))
    @views c = max(zero(T), x[1]*x[1] - dot(x[2:end], x[2:end])) #should be ≥0
    d = b^2 - 4*a*c

    if(c < 0)
        return -Inf
    end

    if( (a > 0 && b > 0) || d < 0)
        return αmax

    elseif a == 0
        return αmax

    elseif c == 0
        return (a >= 0 ? αmax : zero(T)) 
    end 

    t = (b >= 0) ? (-b - sqrt(d)) : (-b + sqrt(d))

    r1 = (2*c)/t;
    r2 = t/(2*a);

    r1 = r1 < 0 ? floatmax(T) : r1
    r2 = r2 < 0 ? floatmax(T) : r2

    return min(αmax,r1,r2)

end


##################################################################
# Functional variants for second-order cones
# (several large socs)
##################################################################
@inline function update_scaling_soc!(
    ::Val{1},
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    workz::AbstractVector{T},
    works::AbstractVector{T},
    workw::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    #initialize kernel for several sparse SOCs
    dummy_int = Cint(1024)
    kernel1 = @cuda launch=false _kernel_parent1_update_scaling(s, z, w, λ, η, workz, works, workw, rng_cones, n_linear, n_sparse_soc, dummy_int)
    kernel2 = @cuda launch=false _kernel_parent2_update_scaling(s, z, w, λ, η, workz, works, workw, rng_cones, n_linear, n_sparse_soc, dummy_int)
    kernel3 = @cuda launch=false _kernel_parent3_update_scaling(s, z, w, λ, η, workz, works, workw, rng_cones, n_linear, n_sparse_soc, dummy_int)
    kernel4 = @cuda launch=false _kernel_parent4_update_scaling(s, z, w, λ, η, workz, works, workw, rng_cones, n_linear, n_sparse_soc, dummy_int)
    kernel5 = @cuda launch=false _kernel_parent5_update_scaling(s, z, w, λ, η, workz, works, workw, rng_cones, n_linear, n_sparse_soc, dummy_int)
    kernel6 = @cuda launch=false _kernel_parent6_update_scaling(s, z, w, λ, η, workz, works, workw, rng_cones, n_linear, n_sparse_soc, dummy_int)

    config = launch_configuration(kernel1.fun)
    maxthreads = Cint(config.threads)
    threads = (1)
    blocks = (n_sparse_soc)

    kernel1(s, z, w, λ, η, workz, works, workw, rng_cones, n_linear, n_sparse_soc, maxthreads; threads, blocks, stream=st)
    kernel2(s, z, w, λ, η, workz, works, workw, rng_cones, n_linear, n_sparse_soc, maxthreads; threads, blocks, stream=st)
    kernel3(s, z, w, λ, η, workz, works, workw, rng_cones, n_linear, n_sparse_soc, maxthreads; threads, blocks, stream=st)
    kernel4(s, z, w, λ, η, workz, works, workw, rng_cones, n_linear, n_sparse_soc, maxthreads; threads, blocks, stream=st)
    kernel5(s, z, w, λ, η, workz, works, workw, rng_cones, n_linear, n_sparse_soc, maxthreads; threads, blocks, stream=st)
    kernel6(s, z, w, λ, η, workz, works, workw, rng_cones, n_linear, n_sparse_soc, maxthreads; threads, blocks, stream=st)

    #update remaining dense SOCs
    update_scaling_soc_dense!(Val(n_dense_soc > 0), s, z, w, λ, η, rng_cones, n_linear, n_sparse_soc, n_dense_soc, st)
end

function _kernel_parent1_update_scaling(
    s::AbstractVector{T}, 
    z::AbstractVector{T}, 
    w::AbstractVector{T}, 
    λ::AbstractVector{T}, 
    η::AbstractVector{T}, 
    workz::AbstractVector{T},
    works::AbstractVector{T},
    workw::AbstractVector{T},
    rng_cones::AbstractVector, 
    n_shift::Cint,
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        cone_start_idx = rng_cones[shift].start
        cone_stop_idx = rng_cones[shift].stop
        len_tid = cone_stop_idx - cone_start_idx + one(Cint)
        #thread-block info for cone-wise operations
        thread = min(len_tid, maxthread)

        # Calculate shared memory size per block
        shmem = thread * sizeof(T)

        #first calculate the scaled vector w
        @cuda threads = thread blocks = 1 shmem = shmem dynamic = true _kernel_child_dot(z, workz, cone_start_idx+one(Cint), cone_stop_idx, tidx)
        @cuda threads = thread blocks = 1 shmem = shmem dynamic = true _kernel_child_dot(s, works, cone_start_idx+one(Cint), cone_stop_idx, tidx)
    end
    return nothing
end

function _kernel_parent2_update_scaling(
    s::AbstractVector{T}, 
    z::AbstractVector{T}, 
    w::AbstractVector{T}, 
    λ::AbstractVector{T}, 
    η::AbstractVector{T}, 
    workz::AbstractVector{T},
    works::AbstractVector{T},
    workw::AbstractVector{T},
    rng_cones::AbstractVector, 
    n_shift::Cint,
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        cone_start_idx = rng_cones[shift].start
        cone_stop_idx = rng_cones[shift].stop
        len_tid = cone_stop_idx - cone_start_idx + one(Cint)

        #thread-block info for cone-wise operations without conic features
        thread_plus = min(len_tid, maxthread)
        block_uniform = cld(len_tid, thread_plus)

        z1_tid = z[rng_cones[shift].start]
        s1_tid = s[rng_cones[shift].start]
        zscale_tid = z1_tid*z1_tid - workz[tidx]
        zscale_tid = zscale_tid > zero(T) ? sqrt(zscale_tid) : zero(T)
        sscale_tid = s1_tid*s1_tid - works[tidx]
        sscale_tid = sscale_tid > zero(T) ? sqrt(sscale_tid) : zero(T)

        #the leading scalar term for W^TW
        η[tidx] = sqrt(sscale_tid/zscale_tid)

        # construct w and normalize
        # CUDA.@sync @. w[rng_i] = s[rng_i]/(sscale) - z[rng_i]/(zscale)
        minus_zscale_inv_tid = -inv(zscale_tid)
        sscale_inv_tid = inv(sscale_tid)
        @cuda threads = thread_plus blocks = block_uniform dynamic = true _kernel_axpby(z, s, w, minus_zscale_inv_tid, sscale_inv_tid, cone_start_idx, cone_stop_idx) 

    end
    return nothing
end

function _kernel_parent3_update_scaling(
    s::AbstractVector{T}, 
    z::AbstractVector{T}, 
    w::AbstractVector{T}, 
    λ::AbstractVector{T}, 
    η::AbstractVector{T}, 
    workz::AbstractVector{T},
    works::AbstractVector{T},
    workw::AbstractVector{T},
    rng_cones::AbstractVector, 
    n_shift::Cint,
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        cone_start_idx = rng_cones[shift].start
        cone_stop_idx = rng_cones[shift].stop
        len_tid = cone_stop_idx - cone_start_idx + one(Cint)
        #thread-block info for cone-wise operations
        thread = min(len_tid, maxthread)

        # Calculate shared memory size per block
        shmem = thread * sizeof(T)

        z1_tid = z[cone_start_idx]
        zscale_tid = z1_tid*z1_tid - workz[tidx]
        zscale_tid = zscale_tid > zero(T) ? sqrt(zscale_tid) : zero(T)

        w[cone_start_idx]  += 2*z[cone_start_idx]/(zscale_tid)

        #     @views workw[tidx] = dot(wi[2:end], wi[2:end])
        @cuda threads = thread blocks = 1 shmem = shmem dynamic = true _kernel_child_dot(w, workw, cone_start_idx+one(Cint), cone_stop_idx, tidx)
    end
    return nothing
end

function _kernel_parent4_update_scaling(
    s::AbstractVector{T}, 
    z::AbstractVector{T}, 
    w::AbstractVector{T}, 
    λ::AbstractVector{T}, 
    η::AbstractVector{T}, 
    workz::AbstractVector{T},
    works::AbstractVector{T},
    workw::AbstractVector{T},
    rng_cones::AbstractVector, 
    n_shift::Cint,
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        cone_start_idx = rng_cones[shift].start
        len_tid = rng_cones[shift].stop - cone_start_idx + one(Cint)

        #thread-block info for cone-wise operations without conic features
        thread_plus = min(len_tid, maxthread)
        block_uniform = cld(len_tid, thread_plus)

        z1_tid = z[cone_start_idx]
        s1_tid = s[cone_start_idx]
        zscale_tid = z1_tid*z1_tid - workz[tidx]
        zscale_tid = zscale_tid > zero(T) ? sqrt(zscale_tid) : zero(T)
        sscale_tid = s1_tid*s1_tid - works[tidx]
        sscale_tid = sscale_tid > zero(T) ? sqrt(sscale_tid) : zero(T)
        w1_tid = w[cone_start_idx]
        wscale_tid = w1_tid*w1_tid - workw[tidx]
        wscale_tid = wscale_tid > zero(T) ? sqrt(wscale_tid) : zero(T)

        #Compute the scaling point λ.   Should satisfy λ = Wz = W^{-T}s
        γ_tid = 0.5 * wscale_tid
        λ[cone_start_idx] = γ_tid * sqrt(sscale_tid*zscale_tid)

        coef = inv(s[cone_start_idx]/sscale_tid + z[cone_start_idx]/zscale_tid + 2*γ_tid)
        coef2 = sqrt(sscale_tid*zscale_tid)*coef
        c1 = coef2*((γ_tid + z[cone_start_idx]/zscale_tid)/sscale_tid)
        c2 = coef2*((γ_tid + s[cone_start_idx]/sscale_tid)/zscale_tid)

        #     CUDA.@sync @. λi[2:end] = c1*si[2:end] +c2*zi[2:end]
        @cuda threads = thread_plus blocks = block_uniform dynamic = true _kernel_axpby(s, z, λ, c1, c2, rng_cones[shift].start+one(Cint), rng_cones[shift].stop) 
        
        #     @views wscale = wi[1]*wi[1] - dot(wi[2:end], wi[2:end])
        #     wscale = wscale > 0.0 ? sqrt(wscale) : zero(T)
        #     CUDA.@sync wi ./= wscale
        @cuda threads = thread_plus blocks = block_uniform dynamic = true _kernel_div_w(w, wscale_tid, rng_cones[shift].start, rng_cones[shift].stop)

    end
    return nothing
end

function _kernel_parent5_update_scaling(
    s::AbstractVector{T}, 
    z::AbstractVector{T}, 
    w::AbstractVector{T}, 
    λ::AbstractVector{T}, 
    η::AbstractVector{T}, 
    workz::AbstractVector{T},
    works::AbstractVector{T},
    workw::AbstractVector{T},
    rng_cones::AbstractVector, 
    n_shift::Cint,
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        cone_start_idx = rng_cones[shift].start
        cone_stop_idx = rng_cones[shift].stop
        len_tid = cone_stop_idx - cone_start_idx + one(Cint)
        #thread-block info for cone-wise operations
        thread = min(len_tid, maxthread)

        # Calculate shared memory size per block
        shmem = thread * sizeof(T)

        #     #try to force badly scaled w to come out normalized
        #     @views w1sq = dot(wi[2:end], wi[2:end])
        @cuda threads = thread blocks = 1 shmem = shmem dynamic = true _kernel_child_dot(w, workw, cone_start_idx+one(Cint), cone_stop_idx, tidx)
    end
    return nothing
end

function _kernel_parent6_update_scaling(
    s::AbstractVector{T}, 
    z::AbstractVector{T}, 
    w::AbstractVector{T}, 
    λ::AbstractVector{T}, 
    η::AbstractVector{T}, 
    workz::AbstractVector{T},
    works::AbstractVector{T},
    workw::AbstractVector{T},
    rng_cones::AbstractVector, 
    n_shift::Cint,
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        cone_start_idx = rng_cones[shift].start

        #     #try to force badly scaled w to come out normalized
        #     @views w1sq = dot(wi[2:end], wi[2:end])
        #     wi[1] = sqrt(1 + w1sq)
        w[cone_start_idx] = sqrt(1 + workw[tidx])
    end
    return nothing
end

function _kernel_div_w(
    w::AbstractVector{T}, 
    wscale_tid::T, 
    start_idx::Cint, 
    end_idx::Cint
) where {T} 
    tid = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x - one(Cint) + start_idx

    if tid <= end_idx
        w[tid] = w[tid]/wscale_tid
    end
    return nothing
end

##########################################################
# update_scaling_soc_sparse!
##########################################################
@inline function update_scaling_soc_sparse!(
    ::Val{1},
    w::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    vut::AbstractVector{T},
    rng_cones::AbstractVector,
    numel_linear::Cint,
    n_shift::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    dummy_int = Cint(1024)
    kernel = @cuda launch=false _kernel_parent_update_scaling_soc_sparse(w, η, d, vut, rng_cones, numel_linear, n_shift, n_sparse_soc, dummy_int)
    config = launch_configuration(kernel.fun)
    threads = (1)
    blocks = (n_sparse_soc)

    kernel(w, η, d, vut, rng_cones, numel_linear, n_shift, n_sparse_soc, Cint(config.threads), st; threads, blocks, stream=st)
    record(ev, st)
end

function _kernel_parent_update_scaling_soc_sparse(
    w::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    vut::AbstractVector{T},
    rng_cones::AbstractVector,
    numel_linear::Cint,
    n_shift::Cint,
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        len_tid = rng_cones[shift].stop - rng_cones[shift].start + one(Cint)

        #thread-block info for cone-wise operations without conic features
        thread_plus = min(len_tid, maxthread)
        block_uniform = cld(len_tid, thread_plus)

        w1_tid = w[rng_cones[shift].start]
        α_tid  = 2*w1_tid
        wsq_tid    = 2*w1_tid*w1_tid - 1
        wsqinv_tid = 1/wsq_tid
        d_tid    = wsqinv_tid / 2
        d[tidx] = d_tid

        u0_tid  = sqrt(wsq_tid - d_tid)
        u1_tid = α_tid/u0_tid
        v1_tid = sqrt( 2*(2 + wsqinv_tid)/(2*wsq_tid - wsqinv_tid))

        minus_η2_tid = -η[tidx]*η[tidx] 
        # u[1] = minus_η2*u0, v[1] = 0
        shift_vu = Cint(2)*(rng_cones[shift].stop - numel_linear - len_tid)
        vut[shift_vu+len_tid+one(Cint)] = minus_η2_tid*u0_tid
        vut[shift_vu+one(Cint)] = zero(T)
        # @views u[2:end] .= minus_η2.*u1.*w[2:end]
        # @views v[2:end] .= minus_η2.*v1.*w[2:end]
        @cuda threads = thread_plus blocks = (2*block_uniform) dynamic = true _kernel_child_update_scaling_soc_sparse(w, vut, minus_η2_tid, u1_tid, v1_tid, shift_vu, rng_cones[shift].start, rng_cones[shift].stop, len_tid, block_uniform)
    end    

    return nothing      
end

function _kernel_child_update_scaling_soc_sparse(
    w::AbstractVector{T}, 
    vut::AbstractVector{T}, 
    minus_η2::T, 
    u1::T, 
    v1::T, 
    shift_vu::Cint, 
    start_idx::Cint, 
    end_idx::Cint,
    len::Cint, 
    block_shift::Cint
) where {T}
    # v part
    if (blockIdx().x <= block_shift)
        tid = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x + shift_vu + one(Cint)   #skip v0
        tid_w = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x + start_idx      #skip w0

        if (tid_w <= end_idx)
            vut[tid] = minus_η2*v1*w[tid_w]
        end
    # u part
    else
        tid = (blockIdx().x - block_shift -one(Cint))*blockDim().x+threadIdx().x + shift_vu + one(Cint) + len #skip u0
        tid_w = (blockIdx().x - block_shift -one(Cint))*blockDim().x+threadIdx().x + start_idx      #skip w0

        if (tid_w <= end_idx)
            vut[tid] = minus_η2*u1*w[tid_w]
        end
    end

    return nothing
end

@inline function get_Hs_soc!(
    ::Val{1},
    Hsblocks::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    dummy_int = Cint(1024)
    kernel = @cuda launch=false _kernel_parent_get_Hs_soc(Hsblocks, η, d, rng_blocks, n_shift, n_sparse_soc, dummy_int)
    config = launch_configuration(kernel.fun)
    threads = (1)
    blocks = (n_sparse_soc)

    kernel(Hsblocks, η, d, rng_blocks, n_shift, n_sparse_soc, Cint(config.threads); threads, blocks, stream=st)

    #update dense socs
    get_Hs_soc_dense!(Val(n_dense_soc > 0), Hsblocks, w, η, rng_cones, rng_blocks, n_shift + n_sparse_soc, n_dense_soc, n_sparse_soc, st)

    record(ev, st)
end

function _kernel_parent_get_Hs_soc(
    Hsblocks::AbstractVector{T}, 
    η::AbstractVector{T}, 
    d::AbstractVector{T}, 
    rng_blocks::AbstractVector, 
    n_shift::Cint, 
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        len_tid = rng_blocks[shift].stop - rng_blocks[shift].start + one(Cint)
        #thread-block info for cone-wise operations
        thread = min(len_tid, maxthread)

        #thread-block info for cone-wise operations without conic features
        thread_plus = min(len_tid, maxthread)
        block_uniform = cld(len_tid, thread_plus)

        η2_tid = η[tidx]*η[tidx]
        Hsblocks[rng_blocks[shift].start] = η2_tid*d[tidx]
        @cuda threads = thread_plus blocks = block_uniform dynamic = true _kernel_child_get_Hs(Hsblocks, η2_tid, rng_blocks[shift].start+one(Cint), rng_blocks[shift].stop)
    end    

    return nothing    
end

function _kernel_child_get_Hs(
    Hsblocks::AbstractVector{T}, 
    η2::T, 
    start_idx::Cint, 
    end_idx::Cint
) where {T}
    tid = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x - one(Cint) + start_idx
    if tid <= end_idx
        Hsblocks[tid] = η2
    end

    return nothing   
end

@inline function mul_Hs_soc!(
    ::Val{1},
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    work::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    #multiply sparse socs
    dummy_int = Cint(1024)
    kernel1 = @cuda launch=false _kernel_parent1_mul_Hs_soc(y, x, w, η, work, rng_cones, n_linear, n_sparse_soc, dummy_int)
    kernel2 = @cuda launch=false _kernel_parent2_mul_Hs_soc(y, x, w, η, work, rng_cones, n_linear, n_sparse_soc, dummy_int)

    config = launch_configuration(kernel1.fun)
    threads = (1)
    blocks = (n_sparse_soc)

    # y = = H^{-1}x = W^TWx
    # where H^{-1} = \eta^{2} (2*ww^T - J)
    kernel1(y, x, w, η, work, rng_cones, n_linear, n_sparse_soc, Cint(config.threads); threads, blocks, stream=st)
    kernel2(y, x, w, η, work, rng_cones, n_linear, n_sparse_soc, Cint(config.threads); threads, blocks, stream=st)

    #multiply dense socs
    η_shift = view(η, (n_sparse_soc+1):n_soc)
    mul_Hs_dense_soc!(Val(n_dense_soc > 0), y, x, w, η_shift, rng_cones, n_linear+n_sparse_soc, n_dense_soc, st)

    record(ev, st)
end

function _kernel_parent1_mul_Hs_soc(
    y::AbstractVector{T}, 
    x::AbstractVector{T}, 
    w::AbstractVector{T}, 
    η::AbstractVector{T}, 
    work::AbstractVector{T}, 
    rng_cones::AbstractVector, 
    n_shift::Cint, 
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        len_tid = rng_cones[shift].stop - rng_cones[shift].start + one(Cint)
        #thread-block info for cone-wise operations
        thread = min(len_tid, maxthread)

        # Calculate shared memory size per block
        shmem = thread * sizeof(T)

        @cuda threads = thread blocks = 1 shmem = shmem dynamic = true _kernel_child_dot(x, w, work, rng_cones[shift].start, rng_cones[shift].stop, tidx)
    end    

    return nothing
end

function _kernel_parent2_mul_Hs_soc(
    y::AbstractVector{T}, 
    x::AbstractVector{T}, 
    w::AbstractVector{T}, 
    η::AbstractVector{T}, 
    work::AbstractVector{T}, 
    rng_cones::AbstractVector, 
    n_shift::Cint, 
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        len_tid = rng_cones[shift].stop - rng_cones[shift].start + one(Cint)

        #thread-block info for cone-wise operations without conic features
        thread_plus = min(len_tid, maxthread)
        block_uniform = cld(len_tid, thread_plus)

        c_tid = 2*work[tidx]
        η2_tid = η[tidx]*η[tidx]
        y[rng_cones[shift].start] = η2_tid*(-x[rng_cones[shift].start] + c_tid*w[rng_cones[shift].start])
        @cuda threads = thread_plus blocks = block_uniform dynamic = true _kernel_child_mul_Hs(y, x, w, c_tid, η2_tid, rng_cones[shift].start+one(Cint), rng_cones[shift].stop)
    end    

    return nothing
end

function _kernel_child_mul_Hs(
    y::AbstractVector{T}, 
    x::AbstractVector{T}, 
    w::AbstractVector{T}, 
    c::T, 
    η2::T, 
    start_idx::Cint, 
    end_idx::Cint
) where {T}
    tid = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x - one(Cint) + start_idx
    if tid <= end_idx
        y[tid] = η2*(x[tid] + c*w[tid])
    end
    return nothing    
end

@inline function affine_ds_soc!(
    ::Val{1},
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    work::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}

    #initialize kernel
    dummy_int = Cint(1024)
    kernel1 = @cuda launch=false _kernel_parent1_affine_ds_soc(ds, λ, work, rng_cones, n_linear, n_sparse_soc, dummy_int)
    kernel2 = @cuda launch=false _kernel_parent2_affine_ds_soc(ds, λ, work, rng_cones, n_linear, n_sparse_soc, dummy_int)

    config = launch_configuration(kernel1.fun)
    threads = (1)
    blocks = (n_sparse_soc)

    kernel1(ds, λ, work, rng_cones, n_linear, n_sparse_soc, Cint(config.threads); threads, blocks, stream=st)
    kernel2(ds, λ, work, rng_cones, n_linear, n_sparse_soc, Cint(config.threads); threads, blocks, stream=st)

    #dense blocks
    affine_ds_dense_soc!(Val(n_dense_soc > 0), ds, λ, rng_cones, n_linear+n_sparse_soc, n_dense_soc, st)
    record(ev, st)
end

function _kernel_parent1_affine_ds_soc(
    ds::AbstractVector{T}, 
    λ::AbstractVector{T}, 
    work::AbstractVector{T}, 
    rng_cones::AbstractVector, 
    n_shift::Cint, 
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        len_tid = rng_cones[shift].stop - rng_cones[shift].start + one(Cint)
        #thread-block info for cone-wise operations
        thread = min(len_tid, maxthread)

        # Calculate shared memory size per block
        shmem = thread * sizeof(T)

        @cuda threads = thread blocks = 1 shmem = shmem dynamic = true _kernel_child_dot(λ, work, rng_cones[shift].start, rng_cones[shift].stop, tidx)
    end
    
    return nothing       
end

function _kernel_parent2_affine_ds_soc(
    ds::AbstractVector{T}, 
    λ::AbstractVector{T}, 
    work::AbstractVector{T}, 
    rng_cones::AbstractVector, 
    n_shift::Cint, 
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        len_tid = rng_cones[shift].stop - rng_cones[shift].start + one(Cint)

        #thread-block info for cone-wise operations without conic features
        thread_plus = min(len_tid, maxthread)
        block_uniform = cld(len_tid, thread_plus)

        ds[rng_cones[shift].start] = work[tidx]
        c_tid = 2*λ[rng_cones[shift].start]
        @cuda threads = thread_plus blocks = block_uniform dynamic = true _kernel_child_affine_ds(ds, λ, c_tid, rng_cones[shift].start+one(Cint), rng_cones[shift].stop)
    end
    
    return nothing       
end

function _kernel_child_affine_ds(
    ds::AbstractVector{T}, 
    λ::AbstractVector{T}, 
    c::T,
    start_idx::Cint, 
    end_idx::Cint
) where {T} 

    tid = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x - one(Cint) + start_idx
    if tid <= end_idx
        ds[tid] = c*λ[tid]
    end
    return nothing
end

@inline function combined_ds_shift_soc!(
    ::Val{1},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    workz::AbstractVector{T}, 
    works::AbstractVector{T}, 
    work::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    σμ::T,
    st::CuStream,
    ev::CuEvent
) where {T}
    #sparse socs
    dummy_int = Cint(1024)
    kernel1 = @cuda launch=false _kernel_parent1_combined_ds_shift_soc(shift, step_z, step_s, w, η, workz, works, work, rng_cones, n_linear, n_sparse_soc, σμ, dummy_int)
    kernel2 = @cuda launch=false _kernel_parent2_combined_ds_shift_soc(shift, step_z, step_s, w, η, workz, works, work, rng_cones, n_linear, n_sparse_soc, σμ, dummy_int)
    kernel3 = @cuda launch=false _kernel_parent3_combined_ds_shift_soc(shift, step_z, step_s, w, η, workz, works, work, rng_cones, n_linear, n_sparse_soc, σμ, dummy_int)
    kernel4 = @cuda launch=false _kernel_parent4_combined_ds_shift_soc(shift, step_z, step_s, w, η, workz, works, work, rng_cones, n_linear, n_sparse_soc, σμ, dummy_int)

    config = launch_configuration(kernel1.fun)
    threads = (1)
    blocks = (n_sparse_soc)

    kernel1(shift, step_z, step_s, w, η, workz, works, work, rng_cones, n_linear, n_sparse_soc, σμ, Cint(config.threads); threads, blocks, stream=st)
    kernel2(shift, step_z, step_s, w, η, workz, works, work, rng_cones, n_linear, n_sparse_soc, σμ, Cint(config.threads); threads, blocks, stream=st)
    kernel3(shift, step_z, step_s, w, η, workz, works, work, rng_cones, n_linear, n_sparse_soc, σμ, Cint(config.threads); threads, blocks, stream=st)
    kernel4(shift, step_z, step_s, w, η, workz, works, work, rng_cones, n_linear, n_sparse_soc, σμ, Cint(config.threads); threads, blocks, stream=st)

    #dense socs
    combined_ds_shift_dense_soc!(Val(n_dense_soc > 0), shift, step_z, step_s, w, η, rng_cones, n_linear+n_sparse_soc, n_dense_soc, σμ, st)
    record(ev, st)
end

function _kernel_parent1_combined_ds_shift_soc(
    shift_vec::AbstractVector{T}, 
    step_z::AbstractVector{T}, 
    step_s::AbstractVector{T}, 
    w::AbstractVector{T}, 
    η::AbstractVector{T}, 
    workz::AbstractVector{T}, 
    works::AbstractVector{T}, 
    work::AbstractVector{T}, 
    rng_cones::AbstractVector, 
    n_shift::Cint, 
    n_sparse_soc::Cint, 
    σμ::T, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        len_tid = rng_cones[shift].stop - rng_cones[shift].start + one(Cint)
        #thread-block info for cone-wise operations
        thread = min(len_tid, maxthread)

        # Calculate shared memory size per block
        shmem = thread * sizeof(T)

        @cuda threads = thread blocks = 2 shmem = shmem dynamic = true _kernel_child_combined_ds_shift_dot(w, step_z, step_s, workz, works, rng_cones[shift].start+one(Cint), rng_cones[shift].stop, tidx)
    end 

    return nothing    
    
end

function _kernel_parent2_combined_ds_shift_soc(
    shift_vec::AbstractVector{T}, 
    step_z::AbstractVector{T}, 
    step_s::AbstractVector{T}, 
    w::AbstractVector{T}, 
    η::AbstractVector{T}, 
    workz::AbstractVector{T}, 
    works::AbstractVector{T}, 
    work::AbstractVector{T}, 
    rng_cones::AbstractVector, 
    n_shift::Cint, 
    n_sparse_soc::Cint, 
    σμ::T, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        len_tid = rng_cones[shift].stop - rng_cones[shift].start + one(Cint)

        #thread-block info for cone-wise operations without conic features
        thread_plus = min(len_tid, maxthread)
        block_uniform = cld(len_tid, thread_plus)

        η_tid = η[tidx]
        w1_tid = w[rng_cones[shift].start]
        stepz1_tid = step_z[rng_cones[shift].start]
        steps1_tid = step_s[rng_cones[shift].start]
        ζz_tid = workz[tidx]
        ζs_tid = works[tidx]
        cz_tid = stepz1_tid + ζz_tid/(1+w1_tid)
        cs_tid = -steps1_tid + ζs_tid/(1+w1_tid)
        step_z[rng_cones[shift].start] = η_tid*(w1_tid*stepz1_tid + ζz_tid)
        step_s[rng_cones[shift].start] = (one(T)/η_tid)*(w1_tid*steps1_tid - ζs_tid)
        
    #     CUDA.@sync @. step_zi[2:end] = η[i]*(tmpz[2:end] + cz*wi[2:end]) 
    #     CUDA.@sync @. step_si[2:end] = (one(T)/η[i])*(tmps[2:end] + cs*wi[2:end])
        @cuda threads = thread_plus blocks = (2*block_uniform) dynamic = true _kernel_child_combined_ds_shift_axpby(w, step_z, step_s, cz_tid, cs_tid, η_tid, rng_cones[shift].start+one(Cint), rng_cones[shift].stop, block_uniform) 
    end

    return nothing    
    
end

function _kernel_parent3_combined_ds_shift_soc(
    shift_vec::AbstractVector{T}, 
    step_z::AbstractVector{T}, 
    step_s::AbstractVector{T}, 
    w::AbstractVector{T}, 
    η::AbstractVector{T}, 
    workz::AbstractVector{T}, 
    works::AbstractVector{T}, 
    work::AbstractVector{T}, 
    rng_cones::AbstractVector, 
    n_shift::Cint, 
    n_sparse_soc::Cint, 
    σμ::T, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        len_tid = rng_cones[shift].stop - rng_cones[shift].start + one(Cint)
        #thread-block info for cone-wise operations
        thread = min(len_tid, maxthread)

        # Calculate shared memory size per block
        shmem = thread * sizeof(T)
        
    #     #shift = W⁻¹Δs ∘ WΔz - σμe  
    #     val = dot(step_si, step_zi)  
    #     shifti[1] = val - σμ 
        @cuda threads = thread blocks = 1 shmem = shmem dynamic = true _kernel_child_dot(step_s, step_z, work, rng_cones[shift].start, rng_cones[shift].stop, tidx)
    end
    
    return nothing    
    
end

function _kernel_parent4_combined_ds_shift_soc(
    shift_vec::AbstractVector{T}, 
    step_z::AbstractVector{T}, 
    step_s::AbstractVector{T}, 
    w::AbstractVector{T}, 
    η::AbstractVector{T}, 
    workz::AbstractVector{T}, 
    works::AbstractVector{T}, 
    work::AbstractVector{T}, 
    rng_cones::AbstractVector, 
    n_shift::Cint, 
    n_sparse_soc::Cint, 
    σμ::T, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        len_tid = rng_cones[shift].stop - rng_cones[shift].start + one(Cint)

        #thread-block info for cone-wise operations without conic features
        thread_plus = min(len_tid, maxthread)
        block_uniform = cld(len_tid, thread_plus)
        
        shift_vec[rng_cones[shift].start] = work[tidx] - σμ

        s0_tid   = step_s[rng_cones[shift].start]
        z0_tid   = step_z[rng_cones[shift].start]
    #     CUDA.@sync @. shifti[2:end] = s0*step_zi[2:end] + z0*step_si[2:end]  
        @cuda threads = thread_plus blocks = block_uniform dynamic = true _kernel_child_combined_ds_shift_circdot(shift_vec, step_z, step_s, z0_tid, s0_tid, rng_cones[shift].start+one(Cint), rng_cones[shift].stop) 
    end
    
    return nothing    
    
end

function _kernel_child_combined_ds_shift_dot(
    w::AbstractVector{T}, 
    step_z::AbstractVector{T},
    step_s::AbstractVector{T}, 
    workz::AbstractVector{T},
    works::AbstractVector{T},     
    start_idx::Cint, 
    end_idx::Cint, 
    idx::Cint
) where {T}
    tid = threadIdx().x
    blockSize = blockDim().x
    blockID = blockIdx().x

    # shared memory for reduction
    shmem = @cuDynamicSharedMem(T, blockSize)

    temp = zero(T)
    i = (tid - one(Cint)) + start_idx
    stride = blockDim().x               # assumed to use only one thread-block
    
    if (blockID == 1)
        while i <= end_idx
            temp += w[i]*step_z[i]
            i += stride
        end
    elseif (blockID == 2)
        while i <= end_idx
            temp += w[i]*step_s[i]
            i += stride
        end
    end

    # Store to shared memory
    shmem[tid] = temp
    sync_threads()
    
    # Block-level reduction
    # Note: the initial offset should be rounded to power of 2
    offset = nextpow(2, blockSize) >> 1
    while offset > 0
        if tid <= offset && tid + offset <= blockSize
            shmem[tid] += shmem[tid + offset]
        end
        sync_threads()
        offset >>= 1
    end

    # Store partial sum to global memory
    if tid == 1
        if (blockID == 1)
            workz[idx] = shmem[1]
        elseif (blockID == 2)
            works[idx] = shmem[1]
        end
    end

    return nothing
end

function _kernel_child_combined_ds_shift_axpby(
    w::AbstractVector{T}, 
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},  
    cz::T,
    cs::T,
    η::T, 
    start_idx::Cint, 
    end_idx::Cint,
    block_shift::Cint           #alternative as the dynamic parallism doesn't support 2-dim blocks
) where {T}
    if (blockIdx().x <= block_shift)
        tid = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x - one(Cint) + start_idx

        if tid <= end_idx
            step_z[tid] = η*(step_z[tid] + cz*w[tid]) 
        end
    else
        tid = (blockIdx().x - block_shift -one(Cint))*blockDim().x+threadIdx().x - one(Cint) + start_idx

        if tid <= end_idx
            step_s[tid] = (one(T)/η)*(step_s[tid] + cs*w[tid])
        end
    end

    return nothing
end

function _kernel_child_combined_ds_shift_circdot(
    shift::AbstractVector{T}, 
    step_z::AbstractVector{T}, 
    step_s::AbstractVector{T}, 
    z0::T, 
    s0::T, 
    start_idx::Cint, 
    end_idx::Cint
) where {T}
    tid = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x - one(Cint) + start_idx
    
    if tid <= end_idx
        shift[tid] = s0*step_z[tid] + z0*step_s[tid]
    end

    return nothing
end

##########################################################
# Δs_from_Δz_offset_soc
##########################################################
@inline function Δs_from_Δz_offset_soc!(
    ::Val{1},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    workz::AbstractVector{T}, 
    workλ::AbstractVector{T}, 
    workw::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint,
    st::CuStream,
    ev::CuEvent
) where {T}
    #sparse socs
    dummy_int = Cint(1024)
    kernel1 = @cuda launch=false _kernel_parent1_Δs_from_Δz_offset_soc(out, ds, z, w, λ, η, workz, workλ, workw, rng_cones, n_linear, n_sparse_soc, dummy_int)
    kernel2 = @cuda launch=false _kernel_parent2_Δs_from_Δz_offset_soc(out, ds, z, w, λ, η, workz, workλ, workw, rng_cones, n_linear, n_sparse_soc, dummy_int)
    config = launch_configuration(kernel1.fun)
    threads = (1)
    blocks = (n_sparse_soc)

    kernel1(out, ds, z, w, λ, η, workz, workλ, workw, rng_cones, n_linear, n_sparse_soc, Cint(config.threads); threads, blocks, stream=st)
    kernel2(out, ds, z, w, λ, η, workz, workλ, workw, rng_cones, n_linear, n_sparse_soc, Cint(config.threads); threads, blocks, stream=st)

    #dense socs
    Δs_from_Δz_offset_dense_soc!(Val(n_dense_soc > 0), out, ds, z, w, λ, η, rng_cones, n_linear+n_sparse_soc, n_dense_soc, st)
    record(ev,st)
end

function _kernel_parent1_Δs_from_Δz_offset_soc(
    out::AbstractVector{T}, 
    ds::AbstractVector{T}, 
    z::AbstractVector{T}, 
    w::AbstractVector{T}, 
    λ::AbstractVector{T}, 
    η::AbstractVector{T}, 
    workz::AbstractVector{T}, 
    workλ::AbstractVector{T}, 
    workw::AbstractVector{T}, 
    rng_cones::AbstractVector, 
    n_shift::Cint, 
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        len_tid = rng_cones[shift].stop - rng_cones[shift].start + one(Cint)
        #thread-block info for cone-wise operations
        thread = min(len_tid, maxthread)

        #thread-block info for cone-wise operations without conic features
        thread_plus = min(len_tid, maxthread)
        block_uniform = cld(len_tid, thread_plus)

        # Calculate shared memory size per block
        shmem = thread * sizeof(T)

        @cuda threads = thread blocks = 3 shmem = shmem dynamic = true _kernel_child_Δs_from_Δz_offset_dot(ds, z, λ, w, workz, workλ, workw, rng_cones[shift].start+one(Cint), rng_cones[shift].stop, tidx)
    end

    return nothing
end

function _kernel_parent2_Δs_from_Δz_offset_soc(
    out::AbstractVector{T}, 
    ds::AbstractVector{T}, 
    z::AbstractVector{T}, 
    w::AbstractVector{T}, 
    λ::AbstractVector{T}, 
    η::AbstractVector{T}, 
    workz::AbstractVector{T}, 
    workλ::AbstractVector{T}, 
    workw::AbstractVector{T}, 
    rng_cones::AbstractVector, 
    n_shift::Cint, 
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        len_tid = rng_cones[shift].stop - rng_cones[shift].start + one(Cint)

        #thread-block info for cone-wise operations without conic features
        thread_plus = min(len_tid, maxthread)
        block_uniform = cld(len_tid, thread_plus)

        η_tid = η[tidx]
        z1_tid = z[rng_cones[shift].start]
        resz_tid = z1_tid*z1_tid - workz[tidx]
        λ1ds1_tid = workλ[tidx]
        w1ds1_tid = workw[tidx]
        λ1_inv_tid = one(T)/(λ[rng_cones[shift].start])

        c1 = (λ[rng_cones[shift].start]*ds[rng_cones[shift].start] - λ1ds1_tid)/resz_tid
        c2 = w1ds1_tid/(1+w[rng_cones[shift].start])

        out[rng_cones[shift].start] = λ1_inv_tid*(c1*z[rng_cones[shift].start] + η_tid*w1ds1_tid)
        @cuda threads = thread_plus blocks = block_uniform dynamic = true _kernel_child_Δs_from_Δz_offset_soc(out, z, ds, w, c1, c2, η_tid, λ1_inv_tid, rng_cones[shift].start+one(Cint), rng_cones[shift].stop) 
    end

    return nothing
end

function _kernel_child_Δs_from_Δz_offset_dot(
    ds::AbstractVector{T}, 
    z::AbstractVector{T}, 
    λ::AbstractVector{T},
    w::AbstractVector{T},
    workz::AbstractVector{T}, 
    workλ::AbstractVector{T}, 
    workw::AbstractVector{T}, 
    start_idx::Cint, 
    end_idx::Cint, 
    idx::Cint
) where {T}
    tid = threadIdx().x
    blockSize = blockDim().x
    blockID = blockIdx().x

    # shared memory for reduction
    shmem = @cuDynamicSharedMem(T, blockSize)

    temp = zero(T)
    i = (tid - one(Cint)) + start_idx                 
    stride = blockDim().x               
    
    if (blockID == 1)
        while i <= end_idx
            temp += z[i]*z[i]
            i += stride
        end
    elseif (blockID == 2)
        while i <= end_idx
            temp += λ[i]*ds[i]
            i += stride
        end
    elseif (blockID == 3)
        while i <= end_idx
            temp += w[i]*ds[i]
            i += stride
        end
    end

    # Store to shared memory
    shmem[tid] = temp
    sync_threads()
    
    # Block-level reduction
    # Note: the initial offset should be rounded to power of 2
    offset = nextpow(2, blockSize) >> 1
    while offset > 0
        if tid <= offset && tid + offset <= blockSize
            shmem[tid] += shmem[tid + offset]
        end
        sync_threads()
        offset >>= 1
    end

    # Store partial sum to global memory
    if tid == 1
        if (blockID == 1)
            workz[idx] = shmem[1]
        elseif (blockID == 2)
            workλ[idx] = shmem[1]
        elseif (blockID == 3)
            workw[idx] = shmem[1]
        end
    end

    return nothing
end

function _kernel_child_Δs_from_Δz_offset_soc(
    out::AbstractVector{T}, 
    z::AbstractVector{T}, 
    ds::AbstractVector{T}, 
    w::AbstractVector{T}, 
    c1::T, 
    c2::T, 
    η::T, 
    λ1_inv::T, 
    start_idx::Cint, 
    end_idx::Cint
) where {T}
    tid = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x - one(Cint) + start_idx

    if tid <= end_idx
        out[tid] = (-c1*z[tid] + η*(ds[tid] + c2*w[tid]))*λ1_inv
    end

    return nothing
end

@inline function step_length_soc(
    ::Val{1},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    worka::AbstractVector{T},
    workb::AbstractVector{T},
    workc::AbstractVector{T},
    α::AbstractVector{T},
    αmax::T,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    n_dense_soc::Cint,
    n_sparse_soc::Cint
) where {T}
    # sparse socs
    αz = _step_length_soc_sparse(z, dz, worka, workb, workc, n_linear, n_sparse_soc, rng_cones, αmax) 
    αs = _step_length_soc_sparse(s, ds, worka, workb, workc, n_linear, n_sparse_soc, rng_cones, αmax) 
    αmax = min(αz, αs)

    #dense socs
    αmax = step_length_dense_soc(Val(n_dense_soc > 0), dz, ds, z, s, α, αmax, rng_cones, n_linear+n_sparse_soc, n_dense_soc)
    
    return αmax
end

function _step_length_soc_sparse(
    x::AbstractVector{T}, 
    dx::AbstractVector{T}, 
    a::AbstractVector{T}, 
    b::AbstractVector{T}, 
    c::AbstractVector{T}, 
    n_shift::Cint,
    n_sparse_soc::Cint, 
    rng_cones::AbstractVector, 
    αmax::T
) where {T}
    #initialize kernel
    dummy_int = Cint(1024)
    kernel1 = @cuda launch=false _kernel_parent1_step_length(x, dx, a, b, c, αmax, n_shift, n_sparse_soc, rng_cones, dummy_int)
    kernel2 = @cuda launch=false _kernel_parent2_step_length(x, dx, a, b, c, αmax, n_shift, n_sparse_soc, rng_cones, dummy_int)
    config = launch_configuration(kernel1.fun)
    threads = (1)
    blocks = (n_sparse_soc)

    #compute a
    kernel1(x, dx, a, b, c, αmax, n_shift, n_sparse_soc, rng_cones, Cint(config.threads); threads, blocks)
    CUDA.@sync kernel2(x, dx, a, b, c, αmax, n_shift, n_sparse_soc, rng_cones, Cint(config.threads); threads, blocks)

    return minimum(a)
end

#parent compute
function _kernel_parent1_step_length(
    x::AbstractVector{T}, 
    dx::AbstractVector{T}, 
    a::AbstractVector{T}, 
    b::AbstractVector{T}, 
    c::AbstractVector{T}, 
    αmax::T,
    n_shift::Cint,
    n_sparse_soc::Cint, 
    rng_cones::AbstractVector, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        len_tid = rng_cones[shift].stop - rng_cones[shift].start + one(Cint)
        thread = min(len_tid, maxthread)

        # Calculate shared memory size per block
        shmem = thread * sizeof(T)

        @cuda threads = thread blocks = 3 shmem = shmem dynamic = true _kernel_child_step_length_dot(x, dx, a, b, c, rng_cones[shift].start, rng_cones[shift].stop, tidx)
    end
    return nothing
end

function _kernel_parent2_step_length(
    x::AbstractVector{T}, 
    dx::AbstractVector{T}, 
    a::AbstractVector{T}, 
    b::AbstractVector{T}, 
    c::AbstractVector{T}, 
    αmax::T,
    n_shift::Cint,
    n_sparse_soc::Cint, 
    rng_cones::AbstractVector, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx
        a[tidx] = _compute_step_soc(x[rng_cones[shift].start], dx[rng_cones[shift].start], a[tidx], b[tidx], c[tidx], αmax)
    end
    return nothing
end

@inline function _compute_step_soc(x::T, y::T, a::T, b::T, c::T, αmax::T) where {T}
    if x >= 0 && y < 0
        αmax = min(αmax,-x/y)
    end

    d = b^2 - 4*a*c

    if(c < 0)
        return -Inf
    end

    if( (a > 0 && b > 0) || d < 0)
        return αmax

    elseif a == 0
        return αmax

    elseif c == 0
        return (a >= 0 ? αmax : zero(T)) 
    end 

    t = (b >= 0) ? (-b - sqrt(d)) : (-b + sqrt(d))

    r1 = (2*c)/t;
    r2 = t/(2*a);

    r1 = r1 < 0 ? floatmax(T) : r1
    r2 = r2 < 0 ? floatmax(T) : r2

    return min(αmax,r1,r2)
end

#child compute
function _kernel_child_step_length_dot(
    x::AbstractVector{T}, 
    y::AbstractVector{T}, 
    a::AbstractVector{T}, 
    b::AbstractVector{T}, 
    c::AbstractVector{T}, 
    start_idx::Cint, 
    end_idx::Cint, 
    idx::Cint
) where {T}
    tid = threadIdx().x
    blockSize = blockDim().x
    blockID = blockIdx().x
    # global_tid = (blockID - one(Cint)) * blockSize + tid

    # shared memory for reduction
    shmem = @cuDynamicSharedMem(T, blockSize)

    temp = zero(T)
    i = tid + start_idx                 # Skip x[1], y[1]
    stride = blockDim().x               # 
    
    if (blockID == 1)
        while i <= end_idx
            temp += y[i]*y[i]
            i += stride
        end
    elseif (blockID == 2)
        while i <= end_idx
            temp += x[i]*y[i]
            i += stride
        end
    elseif (blockID == 3)
        while i <= end_idx
            temp += x[i]*x[i]
            i += stride
        end
    end

    # Store to shared memory
    shmem[tid] = temp
    sync_threads()
    
    # Block-level reduction
    # Note: the initial offset should be rounded to power of 2
    offset = nextpow(2, blockSize) >> 1
    while offset > 0
        if tid <= offset && tid + offset <= blockSize
            shmem[tid] += shmem[tid + offset]
        end
        sync_threads()
        offset >>= 1
    end

    # @views a = y[1]*y[1] - dot(y[2:end], y[2:end]) #NB: could be negative
    # @views b = 2*(x[1]*y[1] - dot(x[2:end],y[2:end]))
    # @views c = max(zero(T), x[1]*x[1] - dot(x[2:end], x[2:end])) #should be ≥0
    # Store partial sum to global memory
    if tid == 1
        if (blockID == 1)
            a[idx] = y[start_idx]*y[start_idx] - shmem[1]
        elseif (blockID == 2)
            b[idx] = (x[start_idx]*y[start_idx] - shmem[1]) * 2
        elseif (blockID == 3)
            c[idx] = max(zero(T), x[start_idx]*x[start_idx] - shmem[1])
        end
    end

    return nothing
end


#######################################################
# basic child operations
#######################################################
# a[idx] = <x[idx], x[idx]>
function _kernel_child_dot(
    x::AbstractVector{T}, 
    a::AbstractVector{T}, 
    start_idx::Cint, 
    end_idx::Cint, 
    idx::Cint
) where {T}
    tid = threadIdx().x
    blockSize = blockDim().x

    # shared memory for reduction
    shmem = @cuDynamicSharedMem(T, blockSize)

    temp = zero(T)
    i = (tid - one(Cint)) + start_idx
    stride = blockDim().x               # assumed to use only one thread-block
    
    while i <= end_idx
        xi = x[i]
        temp += xi*xi
        i += stride
    end

    # Store to shared memory
    shmem[tid] = temp
    sync_threads()
    
    # Block-level reduction
    # Note: the initial offset should be rounded to power of 2
    offset = nextpow(2, blockSize) >> 1
    while offset > 0
        if tid <= offset && tid + offset <= blockSize
            shmem[tid] += shmem[tid + offset]
        end
        sync_threads()
        offset >>= 1
    end

    # Store partial sum to global memory
    if tid == 1
        a[idx] = shmem[1]
    end

    return nothing
end

# a[idx] = <x[idx], y[idx]>
function _kernel_child_dot(
    x::AbstractVector{T}, 
    y::AbstractVector{T},
    a::AbstractVector{T}, 
    start_idx::Cint, 
    end_idx::Cint, 
    idx::Cint
) where {T}
    tid = threadIdx().x
    blockSize = blockDim().x

    # shared memory for reduction
    shmem = @cuDynamicSharedMem(T, blockSize)

    temp = zero(T)
    i = (tid - one(Cint)) + start_idx
    stride = blockDim().x               # assumed to use only one thread-block
    
    while i <= end_idx
        temp += x[i]*y[i]
        i += stride
    end

    # Store to shared memory
    shmem[tid] = temp
    sync_threads()
    
    # Block-level reduction
    # Note: the initial offset should be rounded to power of 2
    offset = nextpow(2, blockSize) >> 1
    while offset > 0
        if tid <= offset && tid + offset <= blockSize
            shmem[tid] += shmem[tid + offset]
        end
        sync_threads()
        offset >>= 1
    end

    # Store partial sum to global memory
    if tid == 1
        a[idx] = shmem[1]
    end

    return nothing
end

# z = a*x + b*y
function _kernel_axpby(
    x::AbstractVector{T}, 
    y::AbstractVector{T},  
    z::AbstractVector{T}, 
    a::T, 
    b::T, 
    start_idx::Cint, 
    end_idx::Cint
) where {T} 
    tid = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x - one(Cint) + start_idx

    if tid <= end_idx
        z[tid] = a*x[tid] + b*y[tid]
    end
    return nothing
end

##################################################################
# Functional variants for second-order cones (Pending)
# (potentially for several huge socs)
##################################################################
@inline function update_scaling_soc_sequential!(
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}

    CUDA.@allowscalar for i in 1:n_sparse_soc
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        @views zi = z[rng_i] 
        @views si = s[rng_i] 
        @views wi = w[rng_i] 
        @views λi = λ[rng_i]

        #first calculate the scaled vector w
        @views zscale = zi[1]*zi[1] - dot(zi[2:end], zi[2:end])
        zscale = zscale > 0.0 ? sqrt(zscale) : zero(T)
        @views sscale = si[1]*si[1] - dot(si[2:end], si[2:end])
        sscale = sscale > 0.0 ? sqrt(sscale) : zero(T)

        #the leading scalar term for W^TW
        η[i] = sqrt(sscale/zscale)

        # construct w and normalize
        CUDA.@sync @. w[rng_i] = s[rng_i]/(sscale) - z[rng_i]/(zscale)
        wi[1]  += 2*zi[1]/(zscale)
    
        @views wscale = wi[1]*wi[1] - dot(wi[2:end], wi[2:end])
        wscale = wscale > 0.0 ? sqrt(wscale) : zero(T)
        CUDA.@sync wi ./= wscale

        #try to force badly scaled w to come out normalized
        @views w1sq = dot(wi[2:end], wi[2:end])
        wi[1] = sqrt(1 + w1sq)

        #Compute the scaling point λ.   Should satisfy λ = Wz = W^{-T}s
        γi = 0.5 * wscale
        λi[1] = γi 

        coef = inv(si[1]/sscale + zi[1]/zscale + 2*γi)
        coef2 = sqrt(sscale*zscale)*coef
        c1 = ((γi + zi[1]/zscale)/sscale)
        c2 = ((γi + si[1]/sscale)/zscale)

        CUDA.@sync @. λi[2:end] = coef2*(c1*si[2:end] +c2*zi[2:end])
        λi[1] *= sqrt(sscale*zscale)
    end
end


@inline function update_scaling_soc_sparse_sequential!(
    w::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    vut::AbstractVector{T},
    rng_cones::AbstractVector,
    numel_linear::Cint,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}
    CUDA.@allowscalar for i in 1:n_sparse_soc
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        len_i = length(rng_i)
        rng_sparse_i = rng_i .- numel_linear
        startidx = 2*(rng_sparse_i.stop - len_i)
        wi = view(w, rng_i)
        vi = view(vut, (startidx+1):(startidx+len_i))
        ui = view(vut, (startidx+len_i+1):(startidx+2*len_i))
        ηi = η[i]

        d[i] = _update_scaling_soc_sparse!(wi,ui,vi,ηi)
    end
end

@inline function get_Hs_soc_sparse_sequential!(
    Hsblocks::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}

    #For sparse form, we are returning here the diagonal D block 
    #from the sparse representation of W^TW, but not the
    #extra two entries at the bottom right of the block.
    #The AbstractVector for s and z (and its views) don't
    #know anything about the 2 extra sparsifying entries
    CUDA.@allowscalar for i in 1:n_sparse_soc
        shift_i = i + n_shift
        rng_block_i = rng_blocks[shift_i]
        Hsblock_i = view(Hsblocks, rng_block_i)
        CUDA.@sync @. Hsblock_i = η[i]^2
        Hsblock_i[1] *= d[i]
    end
end

@inline function mul_Hs_soc_sequential!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}

    CUDA.@allowscalar for i in 1:n_sparse_soc
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        yi = view(y, rng_i)
        wi = view(w, rng_i)
        xi = view(x, rng_i)

        # y = = H^{-1}x = W^TWx
        # where H^{-1} = \eta^{2} (2*ww^T - J)
        @. yi= xi
        c = 2*dot(wi,xi)
        yi[1] = -xi[1]
        η2 = η[i]^2
        CUDA.@sync @. yi = η2*(yi + c*wi)
    end
end

@inline function affine_ds_soc_sequential!(
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}

    CUDA.@allowscalar for i in 1:n_sparse_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        @views dsi = ds[rng_cone_i] 
        @views λi = λ[rng_cone_i] 

        #circ product λ∘λ
        dsi[1] = dot(λi, λi)
        λi0 = λi[1]

        CUDA.@sync dsi[2:end] = 2*λi0*λi[2:end]
    end
end


@inline function combined_ds_shift_soc_sequential!(
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint,
    σμ::T
) where {T}

    CUDA.@allowscalar for i in 1:n_sparse_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        @views step_zi = step_z[rng_cone_i] 
        @views step_si = step_s[rng_cone_i] 
        @views wi = w[rng_cone_i] 
        @views shifti = shift[rng_cone_i] 
    
        #shift vector used as workspace for a few steps 
        tmp = shifti            

        #Δz <- WΔz
        @. tmp = step_zi
       
        @views ζ = dot(wi[2:end], step_zi[2:end])

        c = tmp[1] + ζ/(1+wi[1])
    
        step_zi[1] = η[i]*(wi[1]*tmp[1] + ζ)
    
        CUDA.@sync @. step_zi[2:end] = η[i]*(tmp[2:end] + c*wi[2:end]) 

        #Δs <- W⁻¹Δs
        @. tmp = step_si         
        @views ζ = dot(wi[2:end],step_si[2:end])

        c = -tmp[1] + ζ/(1+wi[1])
    
        step_si[1] = (one(T)/η[i])*(wi[1]*tmp[1] - ζ)
    
        CUDA.@sync @. step_si[2:end] = (one(T)/η[i])*(tmp[2:end] + c*wi[2:end])

        #shift = W⁻¹Δs ∘ WΔz - σμe  
        val = dot(step_si, step_zi)  
        shifti[1] = val - σμ 

        s0   = step_si[1]
        z0   = step_zi[1]
        CUDA.@sync @. shifti[2:end] = s0*step_zi[2:end] + z0*step_si[2:end]     
    end
end


function Δs_from_Δz_offset_soc_sequential!(
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}

    CUDA.@allowscalar for i in 1:n_sparse_soc
        #out = Wᵀ(λ \ ds).  Below is equivalent,
        #but appears to be a little more stable 
        
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]

        outi = view(out, rng_i)
        dsi = view(ds, rng_i)
        zi = view(z, rng_i)
        wi = view(w, rng_i)
        λi = view(λ, rng_i)
        ηi = η[i]
        resz = _soc_residual(zi)

        @views λ1ds1  = dot(λi[2:end],dsi[2:end])
        @views w1ds1  = dot(wi[2:end],dsi[2:end])

        CUDA.@sync outi .= -zi
        outi[1] = zi[1]

        c = λi[1]*dsi[1] - λ1ds1
        CUDA.@sync outi .*= c/resz

        outi[1] += ηi*w1ds1
        CUDA.@sync @views outi[2:end]  .+= ηi*(dsi[2:end] + w1ds1/(1+wi[1]).*wi[2:end])

        CUDA.@sync outi .*= (1/λi[1])
    end
end


@inline function step_length_soc_sequential(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     αmax::T,
     rng_cones::AbstractVector,
     n_shift::Cint,
     n_sparse_soc::Cint
) where {T}

    CUDA.@allowscalar for i in 1:n_sparse_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        @views si = s[rng_cone_i] 
        @views dsi = ds[rng_cone_i] 
        @views zi = z[rng_cone_i] 
        @views dzi = dz[rng_cone_i]         

        αz   = _step_length_soc_component_gpu_2(zi,dzi,αmax)
        αs   = _step_length_soc_component_gpu_2(si,dsi,αmax)
        αmax = min(αmax, αz, αs)
    end

    return αmax
end