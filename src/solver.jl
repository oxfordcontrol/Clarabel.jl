# -------------------------------------
# utility constructor that includes
# both object creation and setup
#--------------------------------------
function Solver(
    P::AbstractMatrix{T},
    c::Vector{T},
    A::AbstractMatrix{T},
    b::Vector{T},
    cone_types::Vector{SupportedCones},
    cone_dims::Vector{Int};
    kwargs...
) where{T}

    s = Solver{T}()
    setup!(s,P,c,A,b,cone_types,cone_dims,kwargs...)
    return s
end

# -------------------------------------
# setup!
# -------------------------------------

function setup!(s,P,c,A,b,cone_types,cone_dims; kwargs...)
    #this allows override of individual settings during setup
    settings_populate!(s.settings, Dict(kwargs))
    setup!(s,P,c,A,b,cone_types,cone_dims)
end

function setup!(s,P,c,A,b,cone_types,cone_dims,settings::Settings)
    #this allows total override of settings during setup
    s.settings = settings
    setup!(s,P,c,A,b,cone_types,cone_dims)
end

# main setup function
function setup!(
    s::Solver{T},
    P::AbstractMatrix{T},
    q::Vector{T},
    A::AbstractMatrix{T},
    b::Vector{T},
    cone_types::Vector{SupportedCones},
    cone_dims::Vector{Int}
) where{T}

    #make this first to create the timers
    s.info    = DefaultInfo()

    @timeit s.info.timer "setup!" begin

        cone_info   = ConeInfo(cone_types,cone_dims)
        s.data      = DefaultProblemData(P,q,A,b,cone_info)
        s.scalings  = DefaultScalings(s.data.n,cone_info,s.settings)
        s.variables = DefaultVariables(s.data.n,cone_info)
        s.residuals = DefaultResiduals(s.data.n,s.data.m)

        #equilibrate problem data immediately on setup.
        #this prevents multiple equlibrations if solve!
        #is called more than once.  Do this before
        #creating kksolver and its factors
        @timeit s.info.timer "equilibrate" begin
            equilibrate!(s.scalings,s.data,s.settings)
        end

        @timeit s.info.timer "kkt init" begin
            s.kktsolver = DefaultKKTSolver(s.data,s.scalings,s.settings)
        end

        # work variables for assembling step direction LHS/RHS
        s.step_rhs  = DefaultVariables(s.data.n,s.scalings.cone_info)
        s.step_lhs  = DefaultVariables(s.data.n,s.scalings.cone_info)

    end

    return s
end


# -------------------------------------
# solve!
# -------------------------------------
function solve!(
    s::Solver{T}
) where{T}

    #various initializations
    info_reset!(s.info)
    iter   = 0
    isdone = false
    timer  = s.info.timer

    #initial residuals and duality gap
    gap       = T(0)
    sigma     = T(0)

    #solver release info, solver config
    #problem dimensions, cone type etc
    print_header(s.info,s.settings,s.data)

    @timeit timer "solve!" begin

        #initialize variables to some reasonable starting point
        @timeit timer "default start" solver_default_start!(s)

        @timeit timer "IP iteration" begin

        #----------
        # main loop
        #----------
        while true

            debug_rescale(s.variables)

            #update the residuals
            #--------------
            residuals_update!(s.residuals,s.variables,s.data)

            #calculate duality gap (scaled)
            #--------------
            μ = calc_mu(s.variables, s.residuals, s.scalings)

            #convergence check and printing
            #--------------
            @timeit timer "check termination" begin
                isdone = check_termination!(
                    s.info,s.data,s.variables,
                    s.residuals,s.scalings,s.settings,
                    iter == s.settings.max_iter
                )
            end
            iter += 1
            disable_timer!(timer)
            @notimeit print_status(s.info,s.settings)
            enable_timer!(timer)
            isdone && break

            #update the scalings
            #--------------
            @timeit timer "NT scaling" scaling_update!(s.scalings,s.variables)

            #update the KKT system and the constant
            #parts of its solution
            #--------------
            @timeit timer "kkt update" kkt_update!(s.kktsolver,s.data,s.scalings)

            #calculate the affine step
            #--------------
            calc_affine_step_rhs!(
                s.step_rhs, s.residuals,
                s.data, s.variables, s.scalings
            )

            @timeit timer "kkt solve" begin
                kkt_solve!(
                    s.kktsolver, s.step_lhs, s.step_rhs,
                    s.variables, s.scalings, s.data, :affine
                )
            end

            #calculate step length and centering parameter
            #--------------
            α = calc_step_length(s.variables,s.step_lhs,s.scalings)
            σ = calc_centering_parameter(α)

            #DEBUG: PJG cap the centering parameter using a heuristic
            #σ = debug_cap_centering_param(iter,σ,μ)
            #@printf("μ = %e, σμ = %e\n", μ, σ*μ)

            #calculate the combined step and length
            #--------------
            calc_combined_step_rhs!(
                s.step_rhs, s.residuals,
                s.data, s.variables, s.scalings,
                s.step_lhs, σ, μ
            )

            @timeit timer "kkt solve" begin
                kkt_solve!(
                    s.kktsolver, s.step_lhs, s.step_rhs,
                    s.variables, s.scalings, s.data, :combined
                )
            end

            #compute final step length and update the current iterate
            #--------------
            @timeit timer "step length" α  = calc_step_length(s.variables,s.step_lhs,s.scalings)
            α *= s.settings.max_step_fraction

            variables_add_step!(s.variables,s.step_lhs,α)

            #record scalar values from this iteration
            info_save_scalars(s.info,μ,α,σ,iter)

        end  #end while
        #----------
        #----------

        end #end IP iteration timer

        variables_finalize!(s.variables, s.scalings, s.info.status)

    end #end solve! timer

    info_finalize!(s.info)
    print_footer(s.info,s.settings)

    return nothing
end


# Mehrotra heuristic
function calc_centering_parameter(α::T) where{T}

    return σ = (1-α)^3
end


function solver_default_start!(s::Solver{T}) where {T}

    #set all scalings to identity (or zero for the zero cone)
    scaling_identity!(s.scalings)
    #Refactor
    kkt_update!(s.kktsolver,s.data,s.scalings)
    #solve for primal/dual initial points via KKT
    kkt_solve_initial_point!(s.kktsolver,s.variables,s.data)
    #fix up (z,s) so that they are in the cone
    variables_shift_to_cone!(s.variables, s.scalings)

    return nothing
end
