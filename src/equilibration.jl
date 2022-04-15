import LinearAlgebra
import Statistics: mean

#Ruiz Equilibration procedure, using same method as in COSMO.jl

function data_equilibrate!(
        data::DefaultProblemData{T},
        cones::ConeSet{T},
        settings::Settings{T}
) where {T}

    equil = data.equilibration

    #if equilibration is disabled, just return.  Note that
    #the default equilibration structure initializes with
    #identity scaling already.
    if(!settings.equilibrate_enable)
        return
    end

	#references to scaling matrices from workspace
	D = equil.D;   d = equil.d
	E = equil.E;   e = equil.e
	c = equil.c

	#use the inverse scalings as work vectors
	Dwork = equil.Dinv
	Ework = equil.Einv
    dwork = equil.dinv
    ework = equil.einv

	#references to problem data
    #note that P may be triu, but it shouldn't matter
	P = data.P; A = data.A
	q = data.q; b = data.b

    scale_min = settings.equilibrate_min_scaling
    scale_max = settings.equilibrate_max_scaling

	#perform scaling operations for a fixed number of steps
	for i = 1:settings.equilibrate_max_iter

		kkt_col_norms!(P, A, dwork, ework)

		limit_scaling!(dwork, scale_min, scale_max)
		limit_scaling!(ework, scale_min, scale_max)

		dwork .= inv.(sqrt.(dwork))
		ework .= inv.(sqrt.(ework))

		# Scale the problem data and update the
		# equilibration matrices
		scale_data!(P, A, q, b, Dwork, Ework, one(T))
		LinearAlgebra.lmul!(Dwork, D)        #D[:,:] = Dwork*D
		LinearAlgebra.lmul!(Ework, E)        #E[:,:] = Ework*E

		# now use the Dwork array to hold the
		# column norms of the newly scaled P
		# so that we can compute the mean
		col_norms!(dwork, P)
		mean_col_norm_P = mean(dwork)
		inf_norm_q      = norm(q, Inf)

		if mean_col_norm_P  != zero(T) && inf_norm_q != zero(T)

			scale_cost = max(inf_norm_q, mean_col_norm_P)
			scale_cost = limit_scaling(scale_cost, scale_min, scale_max)
			ctmp = one(T) / scale_cost

			# scale the penalty terms and overall scaling
			scalarmul!(P, ctmp)
			q       .*= ctmp
			c[]      *= ctmp
		end

	end #end Ruiz scaling loop

	# fix scalings in cones for which elementwise
    # scaling can't be applied
	if cones_rectify_equilibration!(cones, ework, e)
		#only rescale again of some cones were rectified
		scale_data!(P, A, q, b, I, Ework, one(T))
		LinearAlgebra.lmul!(Ework, E)
	end

	#update the inverse scaling data
	@. equil.dinv = one(T) / d
	@. equil.einv = one(T) / e

	return nothing
end


function limit_scaling!(s::AbstractVector{T}, minval::T, maxval::T) where {T}
	@. s = clip(s, minval, maxval, one(T))

	return nothing
end


function limit_scaling(s::T, minval::T, maxval::T) where {T}
	s = clip(s, minval, maxval, one(T))

	return s
end


function scale_data!(
    P::AbstractMatrix{T},
    A::AbstractMatrix{T},
    q::AbstractVector{T},
    b::AbstractVector{T},
    Ds::Union{IdentityMatrix, AbstractMatrix{T}},
    Es::Union{IdentityMatrix, AbstractMatrix{T}},
    cs::T = one(T)
) where {T <: AbstractFloat}

	lrmul!(Ds, P, Ds) # P[:,:] = Ds*P*Ds
	lrmul!(Es, A, Ds) # A[:,:] = Es*A*Ds
	q[:] = Ds * q
	b[:] = Es * b
	if cs != one(T)
		scalarmul!(P,cs)
		q .*= cs
	end
	return nothing
end
