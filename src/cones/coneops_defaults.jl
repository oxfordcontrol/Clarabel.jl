import Statistics: mean

## -------------------------------------------
# Default implementations for cone operations
# --------------------------------------------

# degree of the cone is the same as dimension
# by default.   Degree will be defined differently
# for the zero cone and SOC (0 and 1, respectively)
dim(K::AbstractCone{T}) where {T} = K.dim
degree(K::AbstractCone{T}) where {T} = K.dim


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

# All other operations will throw an error
# if a type specific implementation has been
# defined.   To define a new cone, you must
# define implementations for each function below.

function update_scaling!(
    K::AbstractCone{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
    λ::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

#configure cone internals to provide W = I scaling
function set_identity_scaling!(
    K::AbstractCone{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

function get_diagonal_scaling!(
    K::AbstractCone{T},
    diagW2::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end



# implements x = y ∘ z
function circle_op!(
    K::AbstractCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# implements x = y \ z
function inv_circle_op!(
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

# implements y = αWx + βy
function gemv_W!(
    K::AbstractCone{T},
    is_transpose::Bool,
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
    is_transpose::Bool,
    x::AbstractVector{T},
    y::AbstractVector{T},
    α::T,
    β::T
) where {T}

  error("Incomplete cone operation specification: ",typeof(K))

end

# implements y = W^TW^{-1}x
function mul_WtWinv!(
    K::AbstractCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# implements y = W^TWx
function mul_WtW!(
    K::AbstractCone{T},
    x::AbstractVector{T},
    y::AbstractVector{T}
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


#return maximum allowable step length while remaining in cone
function step_length(
     K::AbstractCone{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     λ::AbstractVector{T}
) where {T}

     error("Incomplete cone operation specification: ",typeof(K))

end
