using Printf

# -------------------------------------
# utility constructor that includes
# both object creation and setup
#--------------------------------------
function Solver(
    P::AbstractMatrix{T},
    c::Vector{T},
    A::AbstractMatrix{T},
    b::Vector{T},
    cones::Vector{<:SupportedCone},
    kwargs...
) where{T <: AbstractFloat}

    s = Solver{T}()
    setup!(s,P,c,A,b,cones,kwargs...)
    return s
end

# -------------------------------------
# setup!
# -------------------------------------


"""
	setup!(solver, P, q, A, b, cones, [settings])

Populates a [`Solver`](@ref) with a cost function defined by `P` and `q`, and one or more conic constraints defined by `A`, `b` and a description of a conic constraint composed of cones whose types and dimensions are specified by `cones.`

The solver will be configured to solve the following optimization problem:

```
min   1/2 x'Px + q'x
s.t.  Ax + s = b, s ∈ K
```

All data matrices must be sparse.   The matrix `P` is assumed to be symmetric and positive semidefinite, and only the upper triangular part is used.

The cone `K` is a composite cone.   To define the cone the user should provide a vector of cone specifications along
with the appropriate dimensional information.   For example, to generate a cone in the nonnegative orthant followed by
a second order cone, use:

```
cones = [Clarabel.NonnegativeConeT(dim_1),
         Clarabel.SecondOrderConeT(dim_2)]
```

If the argument 'cones' is constructed incrementally, the should should initialize it as an empty array of the supertype for all allowable cones, e.g.

```
cones = Clarabel.SupportedCone[]
push!(cones,Clarabel.NonnegativeConeT(dim_1))
...
```

The optional argument `settings` can be used to pass custom solver settings:
```julia
settings = Clarabel.Settings(verbose = true)
setup!(model, P, q, A, b, cones, settings)
```

To solve the problem, you must make a subsequent call to [`solve!`](@ref)
"""
function setup!(s,P,c,A,b,cones,settings::Settings)
    #this allows total override of settings during setup
    s.settings = settings
    setup!(s,P,c,A,b,cones)
end

function setup!(s,P,c,A,b,cones; kwargs...)
    #this allows override of individual settings during setup
    settings_populate!(s.settings, Dict(kwargs))
    setup!(s,P,c,A,b,cones)
end

# main setup function
function setup!(
    s::Solver{T},
    P::AbstractMatrix{T},
    q::Vector{T},
    A::AbstractMatrix{T},
    b::Vector{T},
    cones::Vector{<:SupportedCone},
) where{T}

    #sanity check problem dimensions
    _check_dimensions(P,q,A,b,cones)

    #make this first to create the timers
    s.info    = DefaultInfo{T}()

    @timeit s.timers "setup!" begin

        s.cones  = ConeSet{T}(cones)
        s.data   = DefaultProblemData{T}(P,q,A,b,s.cones)
        s.data.m == s.cones.numel || throw(DimensionMismatch())

        s.variables = DefaultVariables{T}(s.data.n,s.cones)
        s.residuals = DefaultResiduals{T}(s.data.n,s.data.m)

        #equilibrate problem data immediately on setup.
        #this prevents multiple equlibrations if solve!
        #is called more than once.
        @timeit s.timers "equilibration" begin
            data_equilibrate!(s.data,s.cones,s.settings)
        end

        @timeit s.timers "kkt init" begin
            s.kktsystem = DefaultKKTSystem{T}(s.data,s.cones,s.settings)
        end

        # work variables for assembling step direction LHS/RHS
        s.step_rhs  = DefaultVariables{T}(s.data.n,s.cones)
        s.step_lhs  = DefaultVariables{T}(s.data.n,s.cones)

        # a saved copy of the previous iterate
        s.prev_vars = DefaultVariables{T}(s.data.n,s.cones)

        # user facing results go here
        s.solution    = DefaultSolution{T}(s.data.m,s.data.n)

    end

    return s
end

# sanity check problem dimensions passed by user

function _check_dimensions(P,q,A,b,cones)

    n = length(q)
    m = length(b)
    p = sum(cone -> nvars(cone), cones; init = 0)

    m == size(A)[1] || throw(DimensionMismatch("A and b incompatible dimensions."))
    p == m          || throw(DimensionMismatch("Constraint dimensions inconsistent with size of cones."))
    n == size(A)[2] || throw(DimensionMismatch("A and q incompatible dimensions."))
    n == size(P)[1] || throw(DimensionMismatch("P and q incompatible dimensions."))
    size(P)[1] == size(P)[2] || throw(DimensionMismatch("P not square."))

end


# an enum for reporting strategy checkpointing
@enum StrategyCheckpointResult begin 
    Update = 0
    NoUpdate 
    Fail
end


# -------------------------------------
# solve!
# -------------------------------------

"""
	solve!(solver)

Computes the solution to the problem in a `Clarabel.Solver` previously defined in [`setup!`](@ref).
"""
function solve!(
    s::Solver{T}
) where{T}

    #various initializations
    iter   = 0
    isdone = false

    #initial residuals and duality gap
    gap       = T(0)
    sigma     = T(0)

    #solver release info, solver config
    #problem dimensions, cone type etc
    @notimeit begin
        print_banner(s.settings.verbose)
        info_print_configuration(s.info,s.settings,s.data,s.cones)
        info_print_status_header(s.info,s.settings)
    end

    info_reset!(s.info,s.timers)

    @timeit s.timers "solve!" begin

        #initialize variables to some reasonable starting point
        #@timeit_debug timers "default start"
        @timeit s.timers "default start" solver_default_start!(s)

        @timeit s.timers "IP iteration" begin

        #----------
        # main loop
        #----------

        # Initialize the scaling strategy to be PrimalDual
        scaling_strategy = PrimalDual::ScalingStrategy

        while true

            #update the residuals
            #--------------
            residuals_update!(s.residuals,s.variables,s.data)

            #calculate duality gap (scaled)
            #--------------
            μ = variables_calc_mu(s.variables, s.residuals, s.cones)

            #convergence check and printing
            #--------------

            info_update!(
                s.info,s.data,s.variables,
                s.residuals,s.settings,s.timers
            )
            isdone = info_check_termination!(s.info,s.residuals,s.settings,iter)

            # check for termination due to slow progress and update strategy
            if isdone
                (action,scaling_strategy) = _strategy_checkpoint_insufficient_progress(s,scaling_strategy) 
                if action === NoUpdate || action === Fail  
                    break 
                end  # allow continuation if action === Update
            end

            #increment counter here because we only count
            #iterations that produce a KKT update 
            @notimeit info_print_status(s.info,s.settings)
            iter += 1

            #update the scalings
            #--------------
            variables_scale_cones!(s.variables,s.cones,μ,scaling_strategy)


            #update the KKT system and the constant parts of its solution.
            #Keep track of the success of each step that calls KKT
            #--------------
            is_kkt_solve_success = true

            @timeit s.timers "kkt update" begin
            is_kkt_solve_success &=
                kkt_update!(s.kktsystem,s.data,s.cones)
            end

            #calculate the affine step
            #--------------
            variables_affine_step_rhs!(
                s.step_rhs, s.residuals,
                s.variables, s.cones
            )


            @timeit s.timers "kkt solve" begin
            is_kkt_solve_success &=
                kkt_solve!(
                    s.kktsystem, s.step_lhs, s.step_rhs,
                    s.data, s.variables, s.cones, :affine
                )
            end

            # combined step only on affine step success 
            if is_kkt_solve_success

                #calculate step length and centering parameter
                #--------------
                α = solver_get_step_length(s,:affine,scaling_strategy)
                σ = _calc_centering_parameter(α)

                #calculate the combined step and length
                #--------------
                variables_combined_step_rhs!(
                    s.step_rhs, s.residuals,
                    s.variables, s.cones,
                    s.step_lhs, σ, μ
                )

                @timeit s.timers "kkt solve" begin
                is_kkt_solve_success &=
                    kkt_solve!(
                        s.kktsystem, s.step_lhs, s.step_rhs,
                        s.data, s.variables, s.cones, :combined
                    )
                end

            end

            # check for numerical failure and update strategy
            if !is_kkt_solve_success
                info_save_scalars(s.info,μ,zero(T),one(T),iter)
                (action,scaling_strategy) = _strategy_checkpoint_numerical_error(s,scaling_strategy) 
                if action === Update; continue; end 
                if action === Fail; break; end 
            end

            #compute final step length and update the current iterate
            #--------------
            α = solver_get_step_length(s,:combined,scaling_strategy)

            # check for undersized step and update strategy
            (action,scaling_strategy) = _strategy_checkpoint_small_steps(s, α, scaling_strategy)
            if action === Update || action === Fail  
                info_save_scalars(s.info,μ,zero(T),one(T),iter)
                if action === Update; continue; end 
                if action === Fail; break; end 
            end 
            #

            # Copy previous iterate in case the next one is a dud
            info_save_prev_iterate(s.info,s.variables,s.prev_vars)

            variables_add_step!(s.variables,s.step_lhs,α)

            #record scalar values from this iteration
            info_save_scalars(s.info,μ,α,σ,iter)

        end  #end while
        #----------
        #----------

        end #end IP iteration timer

    end #end solve! timer

    info_finalize!(s.info,s.residuals,s.settings,s.timers)  #halts timers
    solution_finalize!(s.solution,s.data,s.variables,s.info,s.settings)

    @notimeit info_print_footer(s.info,s.settings)

    return s.solution
end


function solver_default_start!(s::Solver{T}) where {T}

    # If there are only symmetric cones, use CVXOPT style initilization
    # Otherwise, initialize along central rays

    if (cones_is_symmetric(s.cones))
        #set all scalings to identity (or zero for the zero cone)
        cones_set_identity_scaling!(s.cones)
        #Refactor
        kkt_update!(s.kktsystem,s.data,s.cones)
        #solve for primal/dual initial points via KKT
        kkt_solve_initial_point!(s.kktsystem,s.variables,s.data)
        #fix up (z,s) so that they are in the cone
        variables_shift_to_cone!(s.variables, s.cones)

    else
        asymmetric_init_cone!(s.variables, s.cones)
    end

    return nothing
end


function solver_get_step_length(s::Solver{T},steptype::Symbol,scaling_strategy::ScalingStrategy) where{T}

    # step length to stay within the cones
    α = variables_calc_step_length(
        s.variables, s.step_lhs,
        s.cones, s.settings, steptype, scaling_strategy
    )

    # additional barrier function limits for asymmetric cones
    if (!cones_is_symmetric(s.cones) && steptype == :combined && scaling_strategy == Dual)
        αinit = α
        α = solver_backtrack_step_to_barrier(s,αinit)
    end
    return α
end


# check the distance to the boundary for asymmetric cones
function solver_backtrack_step_to_barrier(
    s::Solver{T}, αinit::T
) where {T}

    backtrack = s.settings.linesearch_backtrack_step
    α = αinit

    for j = 1:50
        barrier = variables_compute_barrier(s.variables,s.step_lhs,α,s.cones)
        if barrier < one(T)
            return α
        else
            α = backtrack*α   #backtrack line search
        end
    end

    return α
end


# Mehrotra heuristic
function _calc_centering_parameter(α::T) where{T}

    return σ = (1-α)^3
end



function _strategy_checkpoint_insufficient_progress(s::Solver{T},scaling_strategy::ScalingStrategy) where {T} 

    if s.info.status == INSUFFICIENT_PROGRESS
        #recover old iterate since "insufficient progress" often 
        #involves actual degradation of results 
        info_reset_to_prev_iterates(s.info,s.variables,s.prev_vars)
    else 
        # there is no problem, so nothing to do
        return (NoUpdate::StrategyCheckpointResult, scaling_strategy)
    end 

    # If problem is asymmetric, we can try to continue with the dual-only strategy
    if !cones_is_symmetric(s.cones) && (scaling_strategy == PrimalDual::ScalingStrategy)
        s.info.status = UNSOLVED
        return (Update::StrategyCheckpointResult, Dual::ScalingStrategy)
    else
        return (Fail::StrategyCheckpointResult, scaling_strategy)
    end

end 


function _strategy_checkpoint_numerical_error(s::Solver{T},scaling_strategy::ScalingStrategy) where {T}

    # If problem is asymmetric, we can try to continue with the dual-only strategy
    if !cones_is_symmetric(s.cones) && (scaling_strategy == PrimalDual::ScalingStrategy)
        return (Update::StrategyCheckpointResult, Dual::ScalingStrategy)
    else
        #out of tricks.  Bail out with an error
        s.info.status = NUMERICAL_ERROR
        return (Fail::StrategyCheckpointResult,scaling_strategy)
    end
    return (NoUpdate::StrategyCheckpointResult,scaling_strategy)
end 


function _strategy_checkpoint_small_steps(s::Solver{T}, α::T, scaling_strategy::ScalingStrategy) where {T}

    if !cones_is_symmetric(s.cones) &&
        scaling_strategy == PrimalDual::ScalingStrategy && α < s.settings.min_switch_step_length
        return (Update::StrategyCheckpointResult, Dual::ScalingStrategy)

    elseif α < s.settings.min_terminate_step_length
        s.info.status = INSUFFICIENT_PROGRESS
        return (Fail::StrategyCheckpointResult,scaling_strategy)
    end

    return (NoUpdate::StrategyCheckpointResult,scaling_strategy)

end 


# printing 

function Base.show(io::IO, solver::Clarabel.Solver{T}) where {T}
    println(io, "Clarabel model with Float precision: $(T)")
end
