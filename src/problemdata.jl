import LinearAlgebra

function DefaultProblemData{T}(
	P::AbstractMatrix{T},
	q::AbstractVector{T},
	A::AbstractMatrix{T},
	b::AbstractVector{T},
	cones::Vector{SupportedCone},
	settings::Settings{T}
) where {T}

	# some caution is required to ensure we take a minimal,
	# but nonzero, number of data copies during presolve steps 
	P_new, q_new, A_new, b_new, cones_new = 
		(nothing,nothing,nothing,nothing,nothing)

	# make a copy if badly formatted.  istriu is very fast
	if !istriu(P)
		P_new = triu(P)  #copies 
		P = P_new 	     #rust borrow
	end 

	# presolve : return nothing if disabled or no reduction
	# --------------------------------------
	presolver = try_presolver(A,b,cones,settings)  

	if !isnothing(presolver)
		(A_new, b_new, cones_new) = presolve(presolver, A, b, cones)
		(A, b, cones) = (A_new, b_new, cones_new)
	end

	# chordal decomposition : return nothing if disabled or no decomp
	# --------------------------------------
	chordal_info = try_chordal_info(A,b,cones,settings)

	if !isnothing(chordal_info)
		(P_new, q_new, A_new, b_new, cones_new) = 
			decomp_augment!(chordal_info, P, q, A, b, settings)
	end 

	# now make sure we have a clean copy of everything if we
	#haven't made one already.   Necessary since we will scale
	# scale the internal copy and don't want to step on the user
	copy_if_nothing(x,y) = isnothing(x) ? deepcopy(y) : x
	P_new = copy_if_nothing(P_new,P)
	q_new = copy_if_nothing(q_new,q)
	A_new = copy_if_nothing(A_new,A)
	b_new = copy_if_nothing(b_new,b)
	cones_new = copy_if_nothing(cones_new,cones)

	#cap entries in b at INFINITY.  This is important 
	#for inf values that were not in a reduced cone
	#this is not considered part of the "presolve", so
	#can always happen regardless of user settings 
	@. b_new .= min(b_new,T(Clarabel.get_infinity()))
	
	#this ensures m is the *reduced* size m
	(m,n) = size(A_new)

	equilibration = DefaultEquilibration{T}(n,m)

	normq = norm(q_new, Inf)
	normb = norm(b_new, Inf)

	DefaultProblemData{T}(
		P_new,q_new,A_new,b_new,cones_new,
		n,m,equilibration,normq,normb,
		presolver,chordal_info)

end

function data_get_normq!(data::DefaultProblemData{T}) where {T}

	if isnothing(data.normq)
		# recover unscaled norm
		dinv = data.equilibration.dinv
		data.normq = norm_inf_scaled(data.q,dinv)
	end
	return data.normq
end 

function data_get_normb!(data::DefaultProblemData{T}) where {T}

	if isnothing(data.normb)
		# recover unscaled norm
		einv = data.equilibration.einv
		data.normb = norm_inf_scaled(data.b,einv)
	end
	return data.normb
end 

function data_clear_normq!(data::DefaultProblemData{T}) where {T}
		data.normq = nothing
end 

function data_clear_normb!(data::DefaultProblemData{T}) where {T}
		data.normb = nothing
end 

#Ruiz Equilibration procedure, using same method as in COSMO.jl
function data_equilibrate!(
        data::DefaultProblemData{T},
        cones::CompositeCone{T},
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
	(c,d,e) = (equil.c,equil.d,equil.e)

	#use the inverse scalings as work vectors
    dwork = equil.dinv
    ework = equil.einv

	#references to problem data
    #note that P may be triu, but it shouldn't matter
    (P,A,q,b) = (data.P,data.A,data.q,data.b)

    scale_min = settings.equilibrate_min_scaling
    scale_max = settings.equilibrate_max_scaling

	#perform scaling operations for a fixed number of steps
	for i = 1:settings.equilibrate_max_iter

		kkt_col_norms!(P, A, dwork, ework)

		#zero rows or columns should not get scaled
		@. dwork = ifelse(dwork == zero(T),one(T),dwork)
		@. ework = ifelse(ework == zero(T),one(T),ework)

		dwork .= inv.(sqrt.(dwork))
		ework .= inv.(sqrt.(ework))

		#bound the cumulative scaling 
		@. dwork = clip(dwork, scale_min/d, scale_max/d)
		@. ework = clip(ework, scale_min/e, scale_max/e)

		# Scale the problem data and update the
		# equilibration matrices
		scale_data!(P, A, q, b, dwork, ework)
		@. d *= dwork
		@. e *= ework

		# now use the Dwork array to hold the
		# column norms of the newly scaled P
		# so that we can compute the mean
		col_norms!(dwork, P)
		mean_col_norm_P = mean(dwork)
		inf_norm_q      = norm(q, Inf)

		if mean_col_norm_P  != zero(T) && inf_norm_q != zero(T)

			scale_cost = max(inf_norm_q, mean_col_norm_P)
			ctmp = one(T) / scale_cost
			ctmp = clip(ctmp, scale_min/c[], scale_max/c[])

			# scale the penalty terms and overall scaling
			scalarmul!(P, ctmp)
			q       .*= ctmp
			c[]      *= ctmp
		end

	end #end Ruiz scaling loop

	# fix scalings in cones for which elementwise
    # scaling can't be applied.   Rectification should 
	# either do nothing or take a convex combination of
	# scalings over a cone, so shouldn't need to check  
	# bounds on the scalings here 
	if rectify_equilibration!(cones, ework, e)
		#only rescale again if some cones were rectified
		scale_data!(P, A, q, b, nothing, ework)
		@. e *= ework
	end

	#update the inverse scaling data
	@. equil.dinv = one(T) / d
	@. equil.einv = one(T) / e

	return nothing
end


function scale_data!(
    P::AbstractMatrix{T},
    A::AbstractMatrix{T},
    q::AbstractVector{T},
    b::AbstractVector{T},
    d::Option{AbstractVector{T}},
    e::AbstractVector{T}
) where {T <: AbstractFloat}

    if(!isnothing(d))
        lrscale!(d, P, d) # P[:,:] = Ds*P*Ds
		lrscale!(e, A, d) # A[:,:] = Es*A*Ds
        @. q *= d
    else
        lscale!(e, A) # A[:,:] = Es*A
    end

	@. b *= e
    return nothing
end


function try_chordal_info(
    A::SparseMatrixCSC{T}, 
    b::Vector{T}, 
    cones::Vector{SupportedCone}, 
    settings::Settings{T}
) where {T}
    
    if !settings.chordal_decomposition_enable 
        return nothing
    end

    # nothing to do if there are no PSD cones
    if !any(c -> isa(c,PSDTriangleConeT), cones)
        return nothing 
    end 

    chordal_info = ChordalInfo(A, b, cones, settings)

    # no decomposition possible 
    if !is_decomposed(chordal_info)
        return nothing
    end

    return chordal_info
end 


function try_presolver(
    A::AbstractMatrix{T}, 
    b::Vector{T}, 
    cones::Vector{SupportedCone}, 
    settings::Settings{T}
) where {T}

    if(!settings.presolve_enable)
        return nothing
    end

    presolver = Presolver{T}(A,b,cones,settings)

    if !is_reduced(presolver)
        return nothing 
    end 

    return presolver 
end