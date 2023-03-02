
function print_banner(verbose::Bool)

    if !verbose return; end
    println("-------------------------------------------------------------")
    @printf("           Clarabel.jl v%s  -  Clever Acronym              \n", version())
    println("                   (c) Paul Goulart                          ")
    println("                University of Oxford, 2022                   ")
    println("-------------------------------------------------------------")
end
function info_print_configuration(
    info::DefaultInfo{T},
    settings::Settings,
    data::DefaultProblemData{T},
    cones::CompositeCone{T}
) where {T}

    if(settings.verbose == false) return end

    @printf("\nproblem:\n")
    @printf("  variables     = %i\n", data.n)
    @printf("  constraints   = %i\n", data.m)
    @printf("  nnz(P)        = %i\n", nnz(data.P))
    @printf("  nnz(A)        = %i\n", nnz(data.A))
    @printf("  cones (total) = %i\n", length(cones))
    print_conedims_by_type(cones, ZeroCone)
    print_conedims_by_type(cones, NonnegativeCone)
    print_conedims_by_type(cones, SecondOrderCone)
    print_conedims_by_type(cones, PSDTriangleCone)
    print_conedims_by_type(cones, PowerCone)
    print_conedims_by_type(cones, ExponentialCone)
    print_settings(settings, T)
    @printf("\n")

    return nothing
end

function info_print_status_header(
    info::DefaultInfo{T},
    settings::Settings,
) where {T}

    if(settings.verbose == false) return end

    #print a subheader for the iterations info
    @printf("%s", "iter    ")
    @printf("%s", "pcost        ")
    @printf("%s", "dcost       ")
    @printf("%s", "gap       ")
    @printf("%s", "pres      ")
    @printf("%s", "dres      ")
    @printf("%s", "k/t       ")
    @printf("%s", " μ       ")
    @printf("%s", "step      ")
    @printf("\n")
    println("---------------------------------------------------------------------------------------------")

    return nothing
end

function info_print_status(
    info::DefaultInfo{T},
    settings::Settings
) where {T}

    if(settings.verbose == false) return end

    @printf("%3d  ", info.iterations)
    @printf("% .4e  ", info.cost_primal)
    @printf("% .4e  ", info.cost_dual)
    @printf("%.2e  ", min(info.gap_abs,info.gap_rel))
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


function info_print_footer(
    info::DefaultInfo{T},
    settings::Settings
) where {T}

    if(settings.verbose == false) return end

    println("---------------------------------------------------------------------------------------------")
    @printf("Terminated with status = %s\n",SolverStatusDict[info.status])
    @printf("solve time = %s\n",TimerOutputs.prettytime(info.solve_time*1e9))

    return nothing
end


function bool_on_off(v::Bool)
    return  v ? "on" : "off"
end



function print_settings(settings::Settings, T::DataType)

    set = settings
    @printf("\nsettings:\n")

    if(set.direct_kkt_solver)
        @printf("  linear algebra: direct / %s, precision: %s\n", set.direct_solve_method, get_precision_string(T))
    end

    @printf("  max iter = %i, time limit = %f,  max step = %.3f\n",
        set.max_iter,
        set.time_limit,
        set.max_step_fraction,
    )
    #
    @printf("  tol_feas = %0.1e, tol_gap_abs = %0.1e, tol_gap_rel = %0.1e,\n",
        set.tol_feas,
        set.tol_gap_abs,
        set.tol_gap_rel
    )

    @printf("  static reg : %s, ϵ1 = %0.1e, ϵ2 = %0.1e\n",
        bool_on_off(set.static_regularization_enable),
        set.static_regularization_constant,
        set.static_regularization_proportional,

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


function print_conedims_by_type(cones::CompositeCone{T}, type) where {T}

    maxlistlen = 5

    #how many of this type of cone?
    count = cones.type_counts[type]

    #skip if there are none of this type
    if count == 0
        return #don't report if none
    end

    nvars = map(K->Clarabel.numel(K), cones[isa.(cones,type)])
    name  = rpad(string(nameof(type))[1:end-4],11)  #drops "Cone" part
    @printf("    : %s = %i, ", name, count)

    if count == 1
        @printf(" numel = %i",nvars[1])

    elseif count <= maxlistlen
        #print them all
        @printf(" numel = (")
        foreach(x->@printf("%i,",x),nvars[1:end-1])
        @printf("%i)",nvars[end])

    else
        #print first (maxlistlen-1) and the final one
        @printf(" numel = (")
        foreach(x->@printf("%i,",x),nvars[1:(maxlistlen-1)])
        @printf("...,%i)",nvars[end])
    end

    @printf("\n")


end
