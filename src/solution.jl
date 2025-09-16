
function solution_post_process!(
	solution::DefaultSolution{T},
	data::Union{DefaultProblemData{T},DefaultProblemDataGPU{T}},
	variables::DefaultVariables{T},
	info::DefaultInfo{T},
	settings::Settings{T}
) where {T}

	solution.status = info.status
	is_infeasible   = status_is_infeasible(info.status)

    if is_infeasible
		solution.obj_val      = NaN
		solution.obj_val_dual = NaN
	else 
		solution.obj_val      = info.cost_primal
		solution.obj_val_dual = info.cost_dual
	end 

	solution.iterations  = info.iterations
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
		reverse_presolve!(data.presolver, solution, variables)
	elseif settings.direct_solve_method in gpu_solver_list && (length(solution.x) != length(variables.x))
		lenx = length(solution.x)
		@. solution.x = variables.x[1:lenx]		#extra entries are slack variables for large second-order cones
		# println("Solve an augmented problem that only returns value x")
	else
		@. solution.x = variables.x
		@. solution.z = variables.z 
		@. solution.s = variables.s 
	end

	return nothing

end

function solution_finalize!(
	solution::DefaultSolution{T},
	info::DefaultInfo{T},
) where {T}
	solution.setup_phase_time  = info.setup_phase_time
	solution.solve_phase_time  = info.solve_phase_time
	solution.solve_time  = info.solve_time
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
    println(io,"Total time: ",TimerOutputs.prettytime(solution.solve_time*1e9), "(setup time = %s\n",TimerOutputs.prettytime(solution.setup_phase_time*1e9), ", solve time = %s\n",TimerOutputs.prettytime(solution.solve_phase_time*1e9),")")

end
