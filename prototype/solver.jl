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
    cone_dims::Vector{Int},
    settings::Settings{T} = Settings{T}()
) where{T}

    s = Solver{T}()
    setup!(s,P,c,A,b,cone_types,cone_dims,settings)
    return s
end

# -------------------------------------
# setup!
# -------------------------------------
function setup!(
    s::Solver{T},
    P::AbstractMatrix{T},
    c::Vector{T},
    A::AbstractMatrix{T},
    b::Vector{T},
    cone_types::Vector{SupportedCones},
    cone_dims::Vector{Int},
    settings::Settings{T} = Settings{T}()
) where{T}

    cone_info   = ConeInfo(cone_types,cone_dims)

    s.settings  = settings
    s.data      = DefaultProblemData(P,c,A,b,cone_info)
    s.scalings  = DefaultScalings(cone_info)
    s.variables = DefaultVariables(s.data.n,cone_info)
    s.residuals = DefaultResiduals(s.data.n,s.data.m)
    s.kktsolver = DefaultKKTSolver(s.data,s.scalings)
    s.info    = DefaultInfo()

    # work variables for assembling step direction LHS/RHS
    s.step_rhs  = DefaultVariables(s.data.n,s.scalings.cone_info)
    s.step_lhs  = DefaultVariables(s.data.n,s.scalings.cone_info)

    return nothing
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

    #initial residuals and duality gap
    gap       = T(0)
    sigma     = T(0)

    #solver release info, solver config
    #problem dimensions, cone type etc
    print_header(s.info,s.settings,s.data)

    #initialize variables to some reasonable starting point
    solver_default_start!(s)

    #----------
    # main loop
    #----------
    while true

        #update the residuals
        #--------------
        residuals_update!(s.residuals,s.variables,s.data)

        #calculate duality gap (scaled)
        #--------------
        μ = calc_mu(s.variables, s.residuals, s.scalings)

        #convergence check and printing
        #--------------
        isdone = check_termination!(
            s.info,s.data,s.variables,
            s.residuals,s.scalings,s.settings,
            iter == s.settings.max_iter
        )
        iter += 1

        print_status(s.info,s.settings)
        isdone && break

        #update the scalings
        #--------------
        scaling_update!(s.scalings,s.variables)

        #update the KKT system and the constant
        #parts of its solution
        #--------------
        kkt_update!(s.kktsolver,s.data,s.scalings)

        #calculate the affine step
        #--------------
        calc_affine_step_rhs!(
            s.step_rhs, s.residuals,
            s.data, s.variables, s.scalings
        )
        kkt_solve!(
            s.kktsolver, s.step_lhs, s.step_rhs,
            s.variables, s.scalings, s.data
        )

        #calculate step length and centering parameter
        #--------------
        α = calc_step_length(s.variables,s.step_lhs,s.scalings)
        σ = calc_centering_parameter(α)

        #calculate the combined step and length
        #--------------
        calc_combined_step_rhs!(
            s.step_rhs, s.residuals,
            s.data, s.variables, s.scalings,
            s.step_lhs, σ, μ
        )
        kkt_solve!(
            s.kktsolver, s.step_lhs, s.step_rhs,
            s.variables, s.scalings, s.data
        )

        #compute final step length and update the current iterate
        #--------------
        α = 0.99*calc_step_length(s.variables,s.step_lhs,s.scalings) #PJG: make tunable
        variables_add_step!(s.variables,s.step_lhs,α)

        #record scalar values from this iteration
        info_save_scalars(s.info,μ,α,σ,iter)

    end

    info_finalize!(s.info)
    variables_finalize!(s.variables,s.info.status)
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
