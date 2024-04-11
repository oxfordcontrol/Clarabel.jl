
function solution_finalize!(
	solution::DefaultSolution{T},
	data::DefaultProblemData{T},
	variables::DefaultVariables{T},
	info::DefaultInfo{T},
	settings::Settings{T}
) where {T}

	is_infeasible = status_is_infeasible(info.status)

    if is_infeasible
		solution.obj_val = NaN
		solution.obj_val_dual = NaN
    end

	solution.status  = info.status
	solution.obj_val = info.cost_primal
	solution.obj_val_dual = info.cost_dual

	solution.iterations  = info.iterations
	solution.solve_time  = info.solve_time
	solution.r_prim 	 = info.res_primal
	solution.r_dual 	 = info.res_dual

	# unscale the variables to get a solution 
	# to the internal problem as we solved it 
	variables_unscale!(variables,data,is_infeasible)

	# unwind the chordal decomp and presolve, in the 
	# reverse of the order in which they were applied
	if !isnothing(data.chordal_info)
		variables = decomp_reverse!(
			data.chordal_info, variables, data.cones, settings)
	end

	if !isnothing(data.presolver) 
		reverse_presolve!(data.presolver,variables, solution)
	else
		@. solution.x = variables.x
		@. solution.z = variables.z 
		@. solution.s = variables.s 
	end

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
