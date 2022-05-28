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
    μ::T
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


function get_WtW_block!(
    K::ZeroCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    #expecting only a diagonal here, and
    #setting it to zero since this is an
    #equality condition
    WtWblock .= zero(T)

    return nothing
end

function affine_ds!(
    K::ZeroCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T}
) where {T}

    x .= zero(T)

end

# implements x = y ∘ z for the zero cone
function circ_op!(
    K::ZeroCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    x .= zero(T)

    return nothing
end

# implements x = λ \ z for the zerocone.
# We treat λ as zero always for this cone
function λ_inv_circ_op!(
    K::ZeroCone{T},
    x::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    x .= zero(T)

    return nothing
end

# implements x = y \ z for the zero cone
function inv_circ_op!(
    K::ZeroCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    x .= zero(T)

    return nothing
end

# place vector into zero cone
function shift_to_cone!(
    K::ZeroCone{T},z::AbstractVector{T}
) where{T}

    z .= zero(T)

    return nothing
end

# unsymmetric initialization
function unsymmetric_init!(
    K::ZeroCone{T},
	s::AbstractVector{T},
    z::AbstractVector{T}
) where{T}

    s .= zero(T)
    z .= zero(T)

    return nothing
end

# implements y = αWx + βy for the zero cone
function gemv_W!(
    K::ZeroCone{T},
    is_transpose::Symbol,
    x::AbstractVector{T},
    y::AbstractVector{T},
    α::T,
    β::T
) where {T}

    #treat W like zero
    y .= β.*y

    return nothing
end

# implements y = αWx + βy for the nn cone
function gemv_Winv!(
    K::ZeroCone{T},
    is_transpose::Symbol,
    x::AbstractVector{T},
    y::AbstractVector{T},
    α::T,
    β::T
) where {T}

  #treat Winv like zero
  y .= β.*y

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

# compute ds in the combined step where λ ∘ (WΔz + W^{-⊤}Δs) = - ds
function combined_ds!(
    K::ZeroCone{T},
    dz::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T 
) where {T}

    tmp = dz                #alias
    dz .= step_z            #copy for safe call to gemv_W
    gemv_W!(K,:N,tmp,step_z,one(T),zero(T))         #Δz <- WΔz
    tmp .= step_s           #copy for safe call to gemv_Winv
    gemv_Winv!(K,:T,tmp,step_s,one(T),zero(T))      #Δs <- W⁻¹Δs
    circ_op!(K,tmp,step_s,step_z)                   #tmp = W⁻¹Δs ∘ WΔz
    add_scaled_e!(K,tmp,-σμ)                        #tmp = W⁻¹Δs ∘ WΔz - σμe

    return nothing
end

# compute the generalized step Wᵀ(λ \ ds)
function Wt_λ_inv_circ_ds!(
    K::ZeroCone{T},
    lz::AbstractVector{T},
    rz::AbstractVector{T},
    rs::AbstractVector{T},
    Wtlinvds::AbstractVector{T}
) where {T} 

    tmp = lz;
    @. tmp = rz  #Don't want to modify our RHS
    λ_inv_circ_op!(K,tmp,rs)                  #tmp = λ \ ds
    gemv_W!(K,:T,tmp,Wtlinvds,one(T),zero(T)) #Wᵀ(λ \ ds) = Wᵀ(tmp)

    return nothing
end

# compute the generalized step of -WᵀWΔz
function WtW_Δz!(
    K::ZeroCone{T},
    lz::AbstractVector{T},
    ls::AbstractVector{T},
    workz::AbstractVector{T}
) where {T}

    gemv_W!(K,:N,lz,workz,one(T),zero(T))    #work = WΔz
    gemv_W!(K,:T,workz,ls,-one(T),zero(T))   #Δs = -WᵀWΔz

end

function step_length(
     K::ZeroCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T}
) where {T}

    #equality constraints allow arbitrary step length
    huge = inv(eps(T))
    return (huge,huge)
end

# no f_sum for Zerocone
function f_sum(
    K::ZeroCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    return T(0)

end

# no need to check Zerocone 
function _check_neighbourhood(
    K::ZeroCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    η::T
) where {T}

    return true

end

function shadow_iterates!(
    K::ZeroCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    st::AbstractVector{T},
    zt::AbstractVector{T},
) where {T}

    return nothing
end