# -------------------------------------
# Zero Cone
# -------------------------------------

degree(K::ZeroCone{T}) where {T} = 0
numel(K::ZeroCone{T}) where {T}  = K.dim

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
    δ .= one(T)
    return false
end

function margins(
    K::ZeroCone{T},
    z::AbstractVector{T},
    pd::PrimalOrDualCone
) where{T}

    #for either primal or dual case we specify infinite 
    #minimum margin and zero total margin.   
    #if we later shift a vector into the zero cone 
    #using scaled_unit_shift!, we just zero it 
    #out regardless of the applied shift anway 
    return (floatmax(T),zero(T))
end

# place vector into zero cone
function scaled_unit_shift!(
    K::ZeroCone{T},
    z::AbstractVector{T},
    α::T,
    pd::PrimalOrDualCone,
) where{T}

    if pd == PrimalCone::PrimalOrDualCone #zero cone
        z .= zero(T)
    else 
        () #Re^n.  do nothing 
    end

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
    return is_scaling_success = true
end

function get_Hs!(
    K::ZeroCone{T},
    Hsblock::AbstractVector{T}
) where {T}

    #expecting only a diagonal here, and
    #setting it to zero since this is an
    #equality condition
    Hsblock .= zero(T)

    return nothing
end

# compute the product y = WᵀWx
function mul_Hs!(
    K::ZeroCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
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
    work::AbstractVector{T},
    z::AbstractVector{T}
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

