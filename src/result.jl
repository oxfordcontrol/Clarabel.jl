
function result_finalize!(
	result::Result{T},
	data::DefaultProblemData{T},
	variables::DefaultVariables{T},
	info::DefaultInfo{T}
) where {T}

	result.status  = info.status
	result.obj_val = info.cost_primal

	#copy internal variables and undo homogenization
	result.x .= variables.x
	result.z .= variables.z
	result.s .= variables.s

    #if we have an infeasible problem, normalize
    #using κ to get an infeasibility certificate.
    #Otherwise use τ to get a solution.
    if(info.status == PRIMAL_INFEASIBLE || info.status == DUAL_INFEASIBLE)
        scaleinv = one(T) / variables.κ
		result.obj_val = NaN
    else
        scaleinv = one(T) / variables.τ
    end

    @. result.x *= scaleinv
    @. result.z *= scaleinv
    @. result.s *= scaleinv

    #undo the equilibration
    d = data.equilibration.d; dinv = data.equilibration.dinv
    e = data.equilibration.e; einv = data.equilibration.einv
    cscale = data.equilibration.c[]

    @. result.x *=  d
    @. result.z *=  e ./ cscale
    @. result.s *=  einv

	result.info.r_prim 	   = info.res_primal
	result.info.r_dual 	   = info.res_dual
	result.info.iter	   = info.iterations
	result.info.solve_time = info.solve_time

	return nothing

end



function Base.show(io::IO, result::Result)
	print(io,">>> Clarabel - Results\nStatus: ")
	if result.status == SOLVED
		result_color = :green
	else
		result_color = :red
	end
	printstyled(io,"$(string(result.status))\n", color = result_color)
	println(io,"Iterations: $(result.info.iter)")
    println(io,"Objective: $(@sprintf("%.4g", result.obj_val))")
    println(io,"Solve time: ",TimerOutputs.prettytime(result.info.solve_time*1e9))

end

function Base.show(io::IO, result::ResultInfo)
	print(io,">>> Clarabel - ResultInfo object ")
	return
end
