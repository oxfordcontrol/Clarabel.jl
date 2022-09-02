# -------------------------------------
# Zero Cone
# -------------------------------------

degree(K::ZeroCone{T}) where {T} = 0

function rectify_equilibration!(
    K::ZeroCone{T},
    δ::AbstractVector{T},
    e::AbstractVector{T}
) where{T}

    #allow elementwise equilibration scaling
    δ .= e
    return false
end

function update_scaling!(
    K::ZeroCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    #nothing to do.
    #This cone acts like λ = 0 everywhere.
    return nothing
end

function set_identity_scaling!(
    K::ZeroCone{T}
) where {T}

    #do nothing.   "Identity" scaling will be zero for equalities
    return nothing
end

function affine_ds!(
    K::ZeroCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T}
) where {T}

    x .= zero(T)
end


function combined_ds_shift!(
    K::ZeroCone{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}

    shift .= zero(T)
    return nothing
end

function get_WtW!(
    K::ZeroCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    #expecting only a diagonal here, and
    #setting it to zero since this is an
    #equality condition
    WtWblock .= zero(T)

    return nothing
end

# implements y = αWx + βy for the zero cone
function mul_W!(
    K::ZeroCone{T},
    is_transpose::Symbol,
    y::AbstractVector{T},
    x::AbstractVector{T},
    α::T,
    β::T
) where {T}

    #treat W like zero
    y .= β.*y

    return nothing
end

# implements y = αWx + βy for the nn cone
function mul_Winv!(
    K::ZeroCone{T},
    is_transpose::Symbol,
    y::AbstractVector{T},
    x::AbstractVector{T},
    α::T,
    β::T
) where {T}

  #treat Winv like zero
  y .= β.*y

  return nothing
end



# compute the product y = c ⋅ WᵀWx
function mul_WtW!(
    K::ZeroCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    c::T,
    work::AbstractVector{T}
) where {T}

    y .= 0

end

# compute the generalized step Wᵀ(λ \ ds)
function Wt_λ_inv_circ_ds!(
    K::ZeroCone{T},
    lz::AbstractVector{T},
    rz::AbstractVector{T},
    rs::AbstractVector{T},
    Wtlinvds::AbstractVector{T}
) where {T}

    lz .= zero(T)

    return nothing
end

# place vector into zero cone
function shift_to_cone!(
    K::ZeroCone{T},z::AbstractVector{T}
) where{T}

    z .= zero(T)

    return nothing
end

# unit initialization for asymmetric solves
function unit_initialization!(
    K::ZeroCone{T},
	z::AbstractVector{T},
    s::AbstractVector{T}
) where{T}

    s .= zero(T)
    z .= zero(T)

    return nothing
end

# implements y = y + αe for the zero cone
function add_scaled_e!(
    K::ZeroCone{T},
    x::AbstractVector{T},α::T
) where {T}

    #e = 0, do nothing
    return nothing

end

function step_length(
     K::ZeroCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     settings::Settings{T},
     αmax::T,
) where {T}

    #equality constraints allow arbitrary step length
    return (αmax,αmax)
end

# no compute_centrality for Zerocone
function compute_barrier(
    K::ZeroCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T
) where {T}

    return zero(T)

end
