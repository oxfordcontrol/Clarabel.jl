## -------------------------------------------
# Default implementations for cone operations
# --------------------------------------------
 
# degree of the cone is the same as dimension
# and numel by default.   Degree is different
# for the zero cone and SOC (0 and 1, respectively)
# For SDP cone, the matrix is in R^{dim \times dim},
# so numel will be different.   The length of (s,z)
# vectors will be \sum_i numel(K_i)
function degree(K::AbstractCone{T}) where {T} 
    error("Incomplete cone operation specification: ",typeof(K))
end 

function numel(K::AbstractCone{T}) where {T} 
    error("Incomplete cone operation specification: ",typeof(K))
end

#All cones assumed to provide only dense Hessians / no expansion unless overridden
is_sparse_expandable(::AbstractCone{T}) where {T} = false

#All cones default to symmetric unless overridden
is_symmetric(::AbstractCone{T}) where {T} = true

#All cones support primal dual scaling unless otherwise specified
allows_primal_dual_scaling(::AbstractCone{T}) where {T} = true

# converts an elementwise scaling into
# a scaling that preserves cone memership
# NB:custom rectify functions should return
# true unless δ == e on return
function rectify_equilibration!(
    K::AbstractCone{T},
    δ::AbstractVector{T},
    e::AbstractVector{T}
) where{T}

    #all cones default to scalar equilibration
    #unless otherwise specified
    tmp    = mean(e)
    @.δ    = tmp / e

    return true
end

# All other operations will throw an error
# if a type specific implementation has been
# defined.   To define a new cone, you must
# define implementations for each function below.

# returns (α,β) such that:
# z - α⋅e is just on the cone boundary, with value
# α >=0 indicates z \in cone, i.e. negative margin ===
# outside of the cone.
#
# β is the sum of the margins that are positive.   For most 
# cones this will just be β = max(0.,α), but for cones that 
# are composites (e.g. the R_n^+), it is the sum of all of 
# the positive margin terms.
function margins(
    K::AbstractCone{T},
    z::AbstractVector{T},
    pd::PrimalOrDualCone
) where{T}

    error("Incomplete cone operation specification: ",typeof(K))

end

function scaled_unit_shift!(
    K::AbstractCone{T},
    z::AbstractVector{T},
    α::T
) where{T}

    error("Incomplete cone operation specification: ",typeof(K))

end

function unit_initialization!(
    K::AbstractCone{T},
	z::AbstractVector{T},
    s::AbstractVector{T}
) where{T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# Compute scaling points 

function set_identity_scaling!(
    K::AbstractCone{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

function update_scaling!(
    K::AbstractCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    #NB: should return bool:  `true` on success.

    error("Incomplete cone operation specification: ",typeof(K))

end

# operations on the Hessian of the centrality condition
# : W^TW for symmmetric cones 
# : μH(s) for nonsymmetric cones  

#All cones have diagonal Hs blocks #unless specifically overridden
function Hs_is_diagonal(
    K::AbstractCone{T}
) where{T}
    return true
end

function get_Hs!(
    K::AbstractCone{T},
    Hsblock::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))
end

function mul_Hs!(
    K::AbstractCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    work::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))
end

# ---------------------------------------------------------
# Linearized centrality condition functions 
#
# For nonsymmetric cones:
# -----------------------
#
# The centrality condition is : s = -μg(z)
#
# The linearized version is : 
#     Δs + μH(z)Δz = -ds = -(affine_ds + combined_ds_shift)
#
# The affine term (computed in affine_ds!) is s
# The shift term is μg(z) plus any higher order corrections 
#
# # To recover Δs from Δz, we can write 
#     Δs = - (ds + μHΔz)
# The "offset" in Δs_from_Δz_offset! is then just ds 
# 
# For symmetric cones: 
# --------------------
# 
# The centrality condition is : (W(z + Δz) ∘ W⁻ᵀ(s + Δs) = μe
#
# The linearized version is :   
#     λ ∘ (WΔz + WᵀΔs) = -ds = - (affine_ds + combined_ds_shift)
#
# The affine term (computed in affine_ds!) is λ ∘ λ
# The shift term is W⁻¹Δs_aff ∘ WΔz_aff - σμe, where the terms  
# Δs_aff an Δz_aff are from the affine KKT solve, i.e. they 
# are the Mehrotra correction terms.
#
# To recover Δs from Δz, we can write 
#     Δs = - ( Wᵀ(λ \ ds) + WᵀW Δz)
# The "offset" in Δs_from_Δz_offset! is then Wᵀ(λ \ ds)
#
# Not that the Δs_from_Δz_offset! function is only needed in the 
# general combined step direction.   In the affine step direction,
# we have the identity Wᵀ(λ \ (λ ∘ λ )) = s.  The symmetric and 
# nonsymmetric cases coincide and offset is taken directly as s. 
#
# ---------------------------------------------------------


function affine_ds!(
    K::AbstractCone{T},
    ds::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

function combined_ds_shift!(
    K::AbstractCone{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

function Δs_from_Δz_offset!(
    K::AbstractCone{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# Find the maximum step length in some search direction
function step_length(
     K::AbstractCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     settings::Settings{T},
  αmax::T
) where {T}

     error("Incomplete cone operation specification: ",typeof(K))

end

# Computes f(s + α⋅ds) + f*(z + α⋅dz) for each cone as in §8.3 of Santiago's thesis
function compute_barrier(
    K::AbstractCone{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# ---------------------------------------------
# operations supported by symmetric cones only 
# ---------------------------------------------

# implements y = αWx + βy
function mul_W!(
    K::AbstractCone{T},
    is_transpose::Symbol,  #:T for transpose, :N otherwise
    y::AbstractVector{T},
    x::AbstractVector{T},
    α::T,
    β::T
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))
end


# implements y = αW^{-1}x + βy
function mul_Winv!(
    K::AbstractCone{T},
    is_transpose::Symbol,  #:T for transpose, :N otherwise
    y::AbstractVector{T},
    x::AbstractVector{T},
    α::T,
    β::T
) where {T}

  error("Incomplete cone operation specification: ",typeof(K))
end

# x = λ \ z
# Included as a special case since q \ z for general q is difficult
# to implement for general q i PSD cone and never actually needed. 
function λ_inv_circ_op!(
    K::AbstractCone{T},
    x::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))
end


# ---------------------------------------------
# Jordan algebra operations for symmetric cones 
# ---------------------------------------------

# implements x = y ∘ z
function circ_op!(
    K::AbstractCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

function inv_circ_op!(
    K::AbstractCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end


