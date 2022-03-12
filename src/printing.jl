
function print_status(
    info::DefaultInfo{T},
    settings::Settings
) where {T}

    if(settings.verbose == false) return end

    @printf("%3d  ", info.iterations)
    @printf("% .4e  ", info.cost_primal)
    @printf("% .4e  ", info.cost_dual)
    @printf("%.2e  ", info.res_primal)
    @printf("%.2e  ", info.res_dual)
    @printf("%.2e  ", info.ktratio)
    @printf("%.2e  ", info.μ)
    if(info.iterations > 0)
        @printf("%.2e  ", info.step_length)
    else
        @printf(" ------   ") #info.step_length
    end

    @printf("\n")

    return nothing
end


function print_header(
    info::DefaultInfo{T},
    settings::Settings,
    data::DefaultProblemData{T}
) where {T}

    if(settings.verbose == false) return end

    println("-------------------------------------------------------------")
    @printf("             Clarabel v%s  -  Clever Acronym              \n", version())
    println("                   (c) Paul Goulart                          ")
    println("                University of Oxford, 2022                   ")
    println("-------------------------------------------------------------")
    @printf("problem: \n")
    @printf("  variables     = %i\n", data.n)
    @printf("  constraints   = %i\n", data.m)
    @printf("  nnz(P)        = %i\n", nnz(data.P))
    @printf("  nnz(A)        = %i\n", nnz(data.A))
    @printf("  cones (total) = %i\n", length(data.cone_info.types))
    #@printf("  : zero        = %i", data.cone_info.type_counts[ZeroConeT])
    print_conedims_by_type(data.cone_info, ZeroConeT)
    #@printf("  : nonnegative = %i", data.cone_info.type_counts[NonnegativeConeT])
    print_conedims_by_type(data.cone_info, NonnegativeConeT)
    #@printf("  : secondorder = %i", data.cone_info.type_counts[SecondOrderConeT])
    print_conedims_by_type(data.cone_info, SecondOrderConeT)
    #@printf("  : PSD         = %i", data.cone_info.type_counts[PSDConeT])
    print_conedims_by_type(data.cone_info, PSDConeT)
    print_settings(settings, T)
    @printf("\n")

    #print a subheader for the iterations info
    @printf("%s", "iter    ")
    @printf("%s", "pcost        ")
    @printf("%s", "dcost       ")
    @printf("%s", "pres      ")
    @printf("%s", "dres      ")
    @printf("%s", "k/t       ")
    @printf("%s", " μ       ")
    @printf("%s", "step      ")
    @printf("\n")
    println("-----------------------------------------------------------------------------------")

    return nothing
end


function bool_on_off(v::Bool)
    return  v ? "on" : "off"
end



function print_settings(settings::Settings, T::DataType)

    set = settings
    @printf("settings:\n")

    if(set.direct_kkt_solver)
        @printf("  linear algebra: direct / %s, precision: %s\n", set.direct_solve_method, get_precision_string(T))
    end

    @printf("  max iter = %i, time limit = %f,  max step = %.3f\n",
        set.max_iter,
        set.time_limit == 0 ? Inf : set.max_time,
        set.max_step_fraction,
    )
    #
    @printf("  tol_feas = %0.1e, tol_abs = %0.1e, tol_rel = %0.1e,\n",
        set.tol_feas,
        set.tol_gap_abs,
        set.tol_gap_rel
    )

    @printf("  static reg : %s, ϵ = %0.1e\n",
        bool_on_off(set.static_regularization_enable),
        set.static_regularization_eps
    )
    #
    @printf("  dynamic reg: %s, ϵ = %0.1e, δ = %0.1e\n",
        bool_on_off(set.dynamic_regularization_enable),
        set.dynamic_regularization_eps,
        set.dynamic_regularization_delta
    )
    @printf("  iter refine: %s, reltol = %0.1e, abstol = %0.1e, \n",
        bool_on_off(set.iterative_refinement_enable),
        set.iterative_refinement_reltol,
        set.iterative_refinement_abstol
    )
    @printf("               max iter = %d, stop ratio = %.1f\n",
        set.iterative_refinement_max_iter,
        set.iterative_refinement_stop_ratio
    )
    @printf("  equilibrate: %s, min_scale = %0.1e, max_scale = %0.1e\n",
        bool_on_off(set.equilibrate_enable),
        set.equilibrate_min_scaling,
        set.equilibrate_max_scaling
    )
    @printf("               max iter = %d\n",
        set.equilibrate_max_iter,
    )

    return nothing
end

get_precision_string(T::Type{<:Real}) = string(T)
get_precision_string(T::Type{<:BigFloat}) = string(T," (", precision(T), " bit)")


function print_conedims_by_type(c::ConeInfo, type::SupportedCones)

    maxlistlen = 5

    #how many of this type of cone?
    count = c.type_counts[type]

    #skip if there are none of this type
    if count == 0
        return #don't report if none
    end

    dims  = c.dims[c.types .== type]
    name  = rpad(string(type)[1:end-5],11)  #drops "ConeT part"
    @printf("    : %s = %i, ", name, count)

    if count == 1
        @printf(" dim = %i",dims[1])

    elseif count <= maxlistlen
        #print them all
        @printf(" dim = (")
        foreach(x->@printf("%i,",x),dims[1:end-1])
        @printf("%i)",dims[end])

    else
        #print first (maxlistlen-1) and the final one
        @printf(" dim = (")
        foreach(x->@printf("%i,",x),dims[1:(maxlistlen-1)])
        @printf("...,%i)",dims[end])
    end

    @printf("\n")


end

function print_footer(
    info::DefaultInfo{T},
    settings::Settings
) where {T}

    if(settings.verbose == false) return end

    println("-----------------------------------------------------------------------------------")
    @printf("Terminated with status = %s\n",SolverStatusDict[info.status])
    @printf("solve time = %s\n",TimerOutputs.prettytime(info.solve_time*1e9))

    return nothing
end
