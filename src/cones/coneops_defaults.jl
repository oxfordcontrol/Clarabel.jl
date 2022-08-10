import Statistics: mean

## -------------------------------------------
# Default implementations for cone operations
# --------------------------------------------

# degree of the cone is the same as dimension
# and numel by default.   Degree is different
# for the zero cone and SOC (0 and 1, respectively)
# For SDP cone, the matrix is in R^{dim \times dim},
# so numel will be different.   The length of (s,z)
# vectors will be \sum_i numel(K_i)
dim(K::AbstractCone{T}) where {T} = K.dim
degree(K::AbstractCone{T}) where {T} = dim(K)
numel(K::AbstractCone{T}) where {T} = dim(K)

#All cones default to symmetric unless overridden
is_symmetric(::AbstractCone{T}) where {T} = true


#NB: custom rectify functions should return
#true unless δ == e on return
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

#All cones have diagonal WtW blocks
#unless specifically overridden
function WtW_is_diagonal(
    K::AbstractCone{T}
) where{T}
    return true
end

# All other operations will throw an error
# if a type specific implementation has been
# defined.   To define a new cone, you must
# define implementations for each function below.

function update_scaling!(
    K::AbstractCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    μ::T,
    flag::Bool
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

#configure cone internals to provide W = I scaling
function set_identity_scaling!(
    K::AbstractCone{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

#populates WtWblock with :
# - the diagonal entries of W^TW, if WtW_is_diagonal(K) == true for this cone
# - the upper triangular entries of W^TW, reported columnwise
#
# Note this function should return W^TW, not -W^TW.  Any change of sign
# required by a linear solver is implemented within the solver object.
function get_WtW_block!(
    K::AbstractCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# returns x = λ∘λ for symmetric cones and x = s for asymmetric cones
# The cone must have an internal mechanism
# for storing the scaled variable λ internally.  This variable
# should be updated at the call to update_scaling!
function affine_ds!(
    K::AbstractCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# implements x = y ∘ z
function circ_op!(
    K::AbstractCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end


# implements x = λ \ z, where λ is the internally
# maintained scaling variable.
function λ_inv_circ_op!(
    K::AbstractCone{T},
    x::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# implements x = y \ z.  Note that this function is
# a more general version of λ_inv_circ_op! and is
# not required directly anywhere by the solver. SOC and
# NN cones (for example) implement this function and then
# call it from λ_inv_circ_op! using their internal scaling
# variable, but it is not compulsory to do it that way.
function inv_circ_op!(
    K::AbstractCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# place vector into the cone
function shift_to_cone!(
    K::AbstractCone{T},
    z::AbstractVector{T}
) where{T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# asymmetric initialization
function asymmetric_init!(
    K::AbstractCone{T},
	s::AbstractVector{T},
    z::AbstractVector{T}
) where{T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# implements y = αWx + βy
function gemv_W!(
    K::AbstractCone{T},
    is_transpose::Symbol,  #:T for transpose, :N otherwise
    x::AbstractVector{T},
    y::AbstractVector{T},
    α::T,
    β::T
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# implements y = αW^{-1}x + βy
function gemv_Winv!(
    K::AbstractCone{T},
    is_transpose::Symbol,  #:T for transpose, :N otherwise
    x::AbstractVector{T},
    y::AbstractVector{T},
    α::T,
    β::T
) where {T}

  error("Incomplete cone operation specification: ",typeof(K))

end

# implements y = y + αe
function add_scaled_e!(
    K::AbstractCone{T},
    x::AbstractVector{T},α::T
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# compute ds in the combined step where λ ∘ (WΔz + W^{-⊤}Δs) = - ds
function combined_ds!(
    K::AbstractCone{T},
    dz::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T,
    scale_flag::Bool
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# compute the generalized step Wᵀ(λ \ ds)
function Wt_λ_inv_circ_ds!(
    K::AbstractCone{T},
    lz::AbstractVector{T},
    rz::AbstractVector{T},
    rs::AbstractVector{T},
    Wtlinvds::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# compute the generalized step of -WᵀWΔz
function WtW_Δz!(
    K::AbstractCone{T},
    lz::AbstractVector{T},
    ls::AbstractVector{T},
    workz::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

#return maximum allowable step length while remaining in cone
#should return a Tuple of allowable step lengths for each direction
#, i.e. (step_z, step_s)
function step_length(
     K::AbstractCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::T,
     backtrack::T
) where {T}

     error("Incomplete cone operation specification: ",typeof(K))

end

# Computes f(s) + f*(z) for each cone as in Section 8.3 in Santiago's thesis
function compute_centrality(
    K::AbstractCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

