
function solution_finalize!(
	solution::DefaultSolution{T},
	data::DefaultProblemData{T},
	variables::DefaultVariables{T},
	info::DefaultInfo{T},
	settings::Settings{T}
) where {T}

	solution.status  = info.status
	solution.obj_val = info.cost_primal

	#copy internal variables and undo homogenization
	solution.x .= variables.x
	solution.z .= variables.z
	solution.s .= variables.s

	# YC: determine the final status if the solver terminates early due to a numerical issue
	if (info.status == EARLY_TERMINATED || info.status == NUMERICALLY_HARD)
		if( ((info.gap_abs < settings.reduced_tol_gap_abs) || (info.gap_rel < settings.reduced_tol_gap_rel))
            && (info.res_primal < settings.reduced_tol_feas)
            && (info.res_dual   < settings.reduced_tol_feas)
        )
			info.status = APPROX_SOLVED
		end
	end

    #if we have an infeasible problem, normalize
    #using κ to get an infeasibility certificate.
    #Otherwise use τ to get a solution.
    if (info.status == PRIMAL_INFEASIBLE || info.status == DUAL_INFEASIBLE)
        scaleinv = one(T) / variables.κ
		solution.obj_val = NaN
    else
        scaleinv = one(T) / variables.τ
    end

    @. solution.x *= scaleinv
    @. solution.z *= scaleinv
    @. solution.s *= scaleinv

    #undo the equilibration
    d = data.equilibration.d; dinv = data.equilibration.dinv
    e = data.equilibration.e; einv = data.equilibration.einv
    cscale = data.equilibration.c[]

    @. solution.x *=  d
    @. solution.z *=  e ./ cscale
    @. solution.s *=  einv

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
    println(io,"Objective: $(@sprintf("%.4g", solution.obj_val))")
    println(io,"Solve time: ",TimerOutputs.prettytime(solution.solve_time*1e9))

end
