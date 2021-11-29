# -----------------------------------------------------
# operators on multiple cones
# -----------------------------------------------------

# Order of the cone is the same as dimension
# by default.   Order will be defined differently
# for the zero cone though (order=0 in that case)
dim(K::AbstractCone{T}) where {T} = K.dim
order(K::AbstractCone{T}) where {T} = K.dim


# x = y ∘ z
function circle_op!(scalings::DefaultConeScalings,
					 x::SplitVector{T},
					 y::SplitVector{T},
					 z::SplitVector{T}) where {T}

    foreach(circle_op!,scalings.cones,x.views,y.views,z.views)

end

# x = y \ z
function inv_circle_op!(scalings::DefaultConeScalings,
					 x::SplitVector{T},
					 y::SplitVector{T},
					 z::SplitVector{T}) where {T}

	foreach(inv_circle_op!,scalings.cones,x.views,y.views,z.views)

end

# place a vector to some nearby point in the cone
function shift_to_cone!(scalings::DefaultConeScalings,z::SplitVector{T}) where {T}

	foreach(shift_to_cone!,scalings.cones,z.views)

end

# computes y = αWx + βy, or y = αWᵀx + βy, i.e.
# similar to the BLAS gemv interface
function gemv_W!(scalings::DefaultConeScalings,
				  is_transpose::Bool,
				  x::SplitVector{T},
				  y::SplitVector{T},
				  α::T,
				  β::T) where {T}

	cones = scalings.cones
    xv    = x.views
    yv    = y.views

	foreach((c,x,y)->gemv_W!(c,is_transpose,x,y,α,β),cones,xv,yv)
end

# computes y = αW^{-1}x + βy, or y = αW⁻ᵀx + βy, i.e.
# similar to the BLAS gemv interface
function gemv_Winv!(scalings::DefaultConeScalings,
				  is_transpose::Bool,
				  x::SplitVector{T},
				  y::SplitVector{T},
				  α::T,
				  β::T) where {T}

	cones = scalings.cones
    xv    = x.views
    yv    = y.views

	foreach((c,x,y)->gemv_Winv!(c,is_transpose,x,y,α,β),cones,xv,yv)
end

#computes y = y + αe
function add_scaled_e!(scalings::DefaultConeScalings,
			  x::SplitVector{T},α::T) where {T}

	foreach((c,x)->add_scaled_e!(c,x,α),scalings.cones,x.views)

end

# maximum allowed step length over all cones
function step_length(scalings::DefaultConeScalings,
					  dz::SplitVector{T},
					  ds::SplitVector{T},
					   z::SplitVector{T},
					   s::SplitVector{T},
					   λ::SplitVector{T}) where {T}

	cones = scalings.cones
	dz    = dz.views
	ds    = ds.views
	z     = z.views
	s     = s.views
	λ     = λ.views

	minimum(map(step_length,cones,dz,ds,z,s,λ))

end



# -------------------------------------
# Zero Cone
# -------------------------------------

order(K::ZeroCone{T}) where {T} = 0


function UpdateScaling!(K::ZeroCone{T},
						s::VectorView{T},
						z::VectorView{T},
						λ::VectorView{T}) where {T}
	λ   .= 0
end

function IdentityScaling!(K::ZeroCone{T}) where {T}
	#do nothing.   "Identity" scaling will be zero for equalities
end


# implements x = y ∘ z for the zero cone
function circle_op!(K::ZeroCone{T},
                    x::VectorView{T},
                    y::VectorView{T},
                    z::VectorView{T}) where {T}
	x .= 0
end

# implements x = y \ z for the zero cone
function inv_circle_op!(K::ZeroCone{T},
                    x::VectorView{T},
                    y::VectorView{T},
                    z::VectorView{T}) where {T}
	x .= 0
end

# place vector into zero cone
function shift_to_cone!(K::ZeroCone{T},z::VectorView{T}) where{T}
	z .= 0
end

# implements y = αWx + βy for the zero cone
function gemv_W!(K::ZeroCone{T},
				  is_transpose::Bool,
				  x::VectorView{T},
				  y::VectorView{T},
				  α::T,
				  β::T) where {T}


 #treat W like zero
 y .= β.*y

end

# implements y = αWx + βy for the nn cone
function gemv_Winv!(K::ZeroCone{T},
				  is_transpose::Bool,
				  x::VectorView{T},
				  y::VectorView{T},
				  α::T,
				  β::T) where {T}
  #treat Winv like zero
  y .= β.*y

end

# implements y = y + αe for the nn cone
function add_scaled_e!(K::ZeroCone{T},
			  		   x::VectorView{T},α::T) where {T}
	#do nothing
end

function make_WTW(K::ZeroCone{T}) where {T}
	#PJG: crazy inefficient
	WTW = spzeros(K.dim,K.dim)
end

function step_length(K::ZeroCone{T},
					 dz::VectorView{T},
					 ds::VectorView{T},
					  z::VectorView{T},
					  s::VectorView{T},
					  λ::VectorView{T}) where {T}

    #equality constraints allow arbitrary step length
	return 1/eps(T)

end



## ------------------------------------
# Nonnegative Cone
# -------------------------------------

NonnegativeCone(args...) = NonnegativeCone{DefaultFloat}(args...)

function UpdateScaling!(K::NonnegativeCone{T},
						s::VectorView{T},
						z::VectorView{T},
						λ::VectorView{T}) where {T}
	λ   .= sqrt.(s.*z)
	K.w .= sqrt.(s./z)
end

function IdentityScaling!(K::NonnegativeCone{T}) where {T}
	K.w .= 1
end


# implements x = y ∘ z for the nn cone
function circle_op!(K::NonnegativeCone{T},
                    x::VectorView{T},
                    y::VectorView{T},
                    z::VectorView{T}) where {T}
	# PJG: note that we call λ ∘ λ at some point, which
	# ends up re-squaring the square root in UpdateScaling!.
	# Maybe could be implemented more efficiently.
	x .= y.*z
end

# implements x = y \ z for the nn cone
function inv_circle_op!(K::NonnegativeCone{T},
                    x::VectorView{T},
                    y::VectorView{T},
                    z::VectorView{T}) where {T}
	x .= z./y
end

# place vector into nn cone
function shift_to_cone!(K::NonnegativeCone{T},z::VectorView{T}) where{T}

	α = minimum(z)
	if(α < eps(T))
		#done in two stages since otherwise (1-α) = -α for
		#large α, which makes z exactly 0. (or worse, -0.0 )
		z .+= -α
		z .+=  1
	end
end

# implements y = αWx + βy for the nn cone
function gemv_W!(K::NonnegativeCone,
				  is_transpose::Bool,
				  x::VectorView{T},
				  y::VectorView{T},
				  α::T,
				  β::T) where {T}


  #W is diagonal, so ignore transposition
  y .= α.*(K.w.*x) + β.*y

end

# implements y = αWx + βy for the nn cone
function gemv_Winv!(K::NonnegativeCone,
				  is_transpose::Bool,
				  x::VectorView{T},
				  y::VectorView{T},
				  α::T,
				  β::T) where {T}


  #W is diagonal, so ignore transposition
  y .= α.*(x./K.w) + β.*y

end

# implements y = y + αe for the nn cone
function add_scaled_e!(K::NonnegativeCone,
			  		   x::VectorView{T},α::T) where {T}
	#e is a vector of ones, so just shift
	x .+= α

end

function make_WTW(K::NonnegativeCone{T}) where {T}

	#PJG: TEMPORARY /  INEFFICIENT
	WTW = SparseMatrixCSC(Diagonal(K.w.^2))

end

function step_length(K::NonnegativeCone{T},
					 dz::VectorView{T},
					 ds::VectorView{T},
					  z::VectorView{T},
					  s::VectorView{T},
					  λ::VectorView{T}) where {T}

	f    = (dv,v)->(dv<0 ? -v/dv : 1/eps(T))
	αz   = minimum(map(f, dz, z))
	αs   = minimum(map(f, ds, s))
	α    = min(αz,αs)

end

# ----------------------------------------------------
# Second Order Cone
# ----------------------------------------------------
