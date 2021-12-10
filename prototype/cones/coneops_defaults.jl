## -------------------------------------------
# Default implementations for cone operations
# --------------------------------------------

# Order of the cone is the same as dimension
# by default.   Order will be defined differently
# for the zero cone though (order=0 in that case)
dim(K::AbstractCone{T}) where {T} = K.dim
order(K::AbstractCone{T}) where {T} = K.dim


# All other operations will throw an error
# if a type specific implementation has been
# defined.   To define a new cone, you must
# define implementations for each function below.

function update_scaling!(
    K::AbstractCone{T},
    s::VectorView{T},
    z::VectorView{T},
    λ::VectorView{T}
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
    diagW2::VectorView{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end



# implements x = y ∘ z
function circle_op!(
    K::AbstractCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# implements x = y \ z
function inv_circle_op!(
    K::AbstractCone{T},
    x::VectorView{T},
    y::VectorView{T},
    z::VectorView{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# place vector into the cone
function shift_to_cone!(
    K::AbstractCone{T},
    z::VectorView{T}
) where{T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# implements y = αWx + βy
function gemv_W!(
    K::AbstractCone{T},
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# implements y = αW^{-1}x + βy
function gemv_Winv!(
    K::AbstractCone{T},
    is_transpose::Bool,
    x::VectorView{T},
    y::VectorView{T},
    α::T,
    β::T
) where {T}

  error("Incomplete cone operation specification: ",typeof(K))

end

# implements y = W^TW^{-1}x
function mul_WtWinv!(
    K::AbstractCone{T},
    x::VectorView{T},
    y::VectorView{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# implements y = W^TWx
function mul_WtW!(
    K::AbstractCone{T},
    x::VectorView{T},
    y::VectorView{T}
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end

# implements y = y + αe
function add_scaled_e!(
    K::AbstractCone{T},
    x::VectorView{T},α::T
) where {T}

    error("Incomplete cone operation specification: ",typeof(K))

end


#return maximum allowable step length while remaining in cone
function step_length(
     K::AbstractCone{T},
    dz::VectorView{T},
    ds::VectorView{T},
     z::VectorView{T},
     s::VectorView{T},
     λ::VectorView{T}
) where {T}

     error("Incomplete cone operation specification: ",typeof(K))

end
