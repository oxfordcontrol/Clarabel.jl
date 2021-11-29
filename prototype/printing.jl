

function PrintStatus(status::DefaultStatus{T},settings::Settings) where {T}

    if(settings.verbose == false) return end

    @printf("%3d  ", status.iterations)
    @printf("% .4e  ", status.cost_primal)
    @printf("% .4e  ", status.cost_dual)
    @printf("%.2e  ", status.res_primal)
    @printf("%.2e  ", status.res_dual)
    @printf("%.2e  ", status.ktratio)
    @printf("%.2e  ", status.gap)
    if(status.iterations > 0)
        @printf("%.2e  ", status.step_length)
        #@printf("%.2e  ", status.sigma)
    else
        @printf(" ------   ") #status.step_length
        #@printf(" ------   ") #status.sigma
    end

    @printf("\n")

end

function PrintHeader(s::Solver{T}) where {T}

    if(s.settings.verbose == false) return end

    println("-----------------------------------------------")
    println("      Clarabel v0.0.0  -  Clever Acronym       ")
    println("            (c) Paul Goulart                   ")
    println("         University of Oxford, 2021            ")
    println("-----------------------------------------------")
    println("problem:")
    @printf("variables   n = %i\n", s.data.n)
    @printf("constraints m = %i\n", s.data.m)
    @printf("cones         = %i\n", length(s.data.cone_info.types))
    @printf("nnz(A)        = %i\n", nnz(s.data.A))
    # PJG: include cone constraint info
    @printf("settings = \n")
    dump(s.settings)
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
    #@printf("%s", "sigma     ")
    @printf("\n")
    println("-----------------------------------------------------------------------------------")

end

function PrintFooter(status::DefaultStatus{T},settings::Settings) where {T}

    if(settings.verbose == false) return end

    println("-----------------------------------------------------------------------------------")
    @printf("Terminated with status = %s\n",SolverStatusDict[status.status])
    @printf("solve time = %s\n",status.solve_time)
end
