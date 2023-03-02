
function solution_finalize!(
	solution::DefaultSolution{T},
	data::DefaultProblemData{T},
	variables::DefaultVariables{T},
	info::DefaultInfo{T},
	settings::Settings{T}
) where {T}

	solution.status  = info.status
	solution.obj_val = info.cost_primal

    #if we have an infeasible problem, normalize
    #using κ to get an infeasibility certificate.
    #Otherwise use τ to get a solution.
    if status_is_infeasible(info.status)
        scaleinv = one(T) / variables.κ
		solution.obj_val = NaN
    else
        scaleinv = one(T) / variables.τ
    end

	#also undo the equilibration
	d = data.equilibration.d; dinv = data.equilibration.dinv
	e = data.equilibration.e; einv = data.equilibration.einv
	cscale = data.equilibration.c[]

	if !is_reduced(data.presolver)
		@. solution.x = d * variables.x * scaleinv
		@. solution.z = e * variables.z * (scaleinv / cscale)
		@. solution.s = einv * variables.s * scaleinv
	else 
		map = data.presolver.lift_map
		@. solution.x = d * variables.x * scaleinv
		@. solution.z[map] = e * variables.z * (scaleinv / cscale)
		@. solution.s[map] = einv * variables.s * scaleinv

		#eliminated constraints get huge slacks 
		#and are assumed to be nonbinding 
		@. solution.s[!data.presolver.reduce_idx] = T(Clarabel.get_infinity())
		@. solution.z[!data.presolver.reduce_idx] = zero(T)
	end 

	solution.iterations  = info.iterations
	solution.solve_time  = info.solve_time
	solution.r_prim 	   = info.res_primal
	solution.r_dual 	   = info.res_dual

	return nothing

end



function Base.show(io::IO, solution::DefaultSolution)
	print(io,">>> Clarabel - Results\nStatus: ")
	if solution.status == SOLVED
		status_color = :green
	else
		status_color = :red
	end
	printstyled(io,"$(string(solution.status))\n", color = status_color)
	println(io,"Iterations: $(solution.iterations)")
    println(io,"Objective: $(@sprintf("%#.4g", solution.obj_val))")
    println(io,"Solve time: ",TimerOutputs.prettytime(solution.solve_time*1e9))

end
