# -------------------------------------
# Zero Cone
# -------------------------------------

degree(K::ZeroCone{T}) where {T} = 0

# The Zerocone reports itself as symmetric even though it is not,
# nor does it support any of the specialised symmetric interface.
# This cone serves as a dummy constraint to allow us to avoid 
# implementing special handling of equalities. We want problems 
# with both equalities and purely symmetric conic constraints to 
# be treated as symmetric for the purposes of initialization etc 
is_symmetric(::ZeroCone{T}) where {T} = true

function rectify_equilibration!(
    K::ZeroCone{T},
    δ::AbstractVector{T},
    e::AbstractVector{T}
) where{T}

    #allow elementwise equilibration scaling
    δ .= e
    return false
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

function set_identity_scaling!(
    K::ZeroCone{T}
) where {T}

    #do nothing.   "Identity" scaling will be zero for equalities
    return nothing
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

function affine_ds!(
    K::ZeroCone{T},
    ds::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

    ds .= zero(T)
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

function Δs_from_Δz_offset!(
    K::ZeroCone{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T}
) where {T}

    out .= zero(T)

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

