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

# functions relating to unit vectors and cone initialization

function shift_to_cone!(
    K::AbstractCone{T},
    z::AbstractVector{T}
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

    error("Incomplete cone operation specification: ",typeof(K))

end

# operations on the Hessian of the centrality condition
# : W^TW for symmmetric cones 
# : μH(s) for nonsymmetric cones  

#All cones have diagonal WtW blocks #unless specifically overridden
function WtW_is_diagonal(
    K::AbstractCone{T}
) where{T}
    return true
end

function get_WtW!(
    K::AbstractCone{T},
    WtWblock::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))
end

function mul_WtW!(
    K::AbstractCone{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    c::T,
    work::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))
end

# PJG : need to complete equation documentation here 
# Residual for centrality condition in step equations 
# For symmetric cones: ds = λ∘λ 
# For nonsymmetric cones: ds = s 
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

# PJG: This needs a new name 
# compute the generalized step Wᵀ(λ \ ds)
function Wt_λ_inv_circ_ds!(
    K::AbstractCone{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T},
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
     α::T
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

# Add the scaled identity element e
function add_scaled_e!(
    K::AbstractCone{T},
    x::AbstractVector{T},α::T
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

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


