
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
    @printf("%.2e  ", info.gap)
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

    println("-----------------------------------------------")
    println("      Clarabel v0.0.0  -  Clever Acronym       ")
    println("            (c) Paul Goulart                   ")
    println("         University of Oxford, 2021            ")
    println("-----------------------------------------------")
    println("problem:")
    @printf("variables     = %i, ", data.n)
    @printf("constraints   = %i\n", data.m)
    @printf("nnz(A)        = %i, ", nnz(data.A))
    @printf("cones         = %i\n", length(data.cone_info.types))
    @printf(": zero        = %i\n", (data.cone_info.k_zerocone))
    @printf(": nonnegative = %i\n", (data.cone_info.k_nncone))
    @printf(": secondorder = %i\n", (data.cone_info.k_socone))
    @printf("settings = \n")
    dump(settings)
    @printf("\n")

    #print a subheader for the iterations info
    @printf("%s", "iter    ")
    @printf("%s", "pcost        ")
    @printf("%s", "dcost       ")
    @printf("%s", "pres      ")
    @printf("%s", "dres      ")
    @printf("%s", "k/t       ")
    @printf("%s", "gap       ")
    @printf("%s", "step      ")
    @printf("\n")
    println("-----------------------------------------------------------------------------------")

    return nothing
end

function print_footer(
    info::DefaultInfo{T},
    settings::Settings
) where {T}

    if(settings.verbose == false) return end

    println("-----------------------------------------------------------------------------------")
    @printf("Terminated with status = %s\n",SolverStatusDict[info.status])
    @printf("solve time = %s\n",info.solve_time)

    return nothing
end
