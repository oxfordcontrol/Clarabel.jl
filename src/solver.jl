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
        s.work_vars = DefaultVariables{T}(s.data.n,s.cones)

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

        scaling_strategy = PrimalDual::ScalingStrategy

        while true

            #update the residuals
            #--------------
            residuals_update!(s.residuals,s.variables,s.data)

            #calculate duality gap (scaled)
            #--------------
            μ = calc_mu(s.variables, s.residuals, s.cones)

            #convergence check and printing
            #--------------
            begin
                info_update!(
                    s.info,s.data,s.variables,
                    s.residuals,s.settings,s.timers
                )
                isdone = info_check_termination!(s.info,s.residuals,s.settings,iter)
            end

            # YC: use the previous iterate as the final solution
            if isdone && s.info.status == EARLY_TERMINATED
                info_reset_to_prev_iterates(s.info,s.variables,s.work_vars)
                break
            end

            iter += 1
            @notimeit info_print_status(s.info,s.settings)
            isdone && break

            #update the scalings
            #--------------
            variables_scale_cones!(s.variables,s.cones,μ,scaling_strategy)

            #calculate the affine step
            #--------------
            calc_affine_step_rhs!(
                s.step_rhs, s.residuals,
                s.variables, s.cones
            )

            #update the KKT system and the constant parts of its solution.  
            #Keep track of the success of each step that calls KKT 
            #--------------
            is_kkt_solve_success = true 
            @timeit s.timers "kkt update" begin 
            is_kkt_solve_success &= 
                kkt_update!(s.kktsystem,s.data,s.cones)
            end 

            @timeit s.timers "kkt solve" begin
            is_kkt_solve_success &= 
                kkt_solve!(
                    s.kktsystem, s.step_lhs, s.step_rhs,
                    s.data, s.variables, s.cones, :affine
                )
            end

            if is_kkt_solve_success 

                #calculate step length and centering parameter
                #--------------
                α = calc_step_length(
                    s.variables, s.step_lhs, s.work_vars,
                    s.cones, s.settings, :affine, scaling_strategy
                )
                σ = calc_centering_parameter(α)
            

                #calculate the combined step and length
                #--------------
                calc_combined_step_rhs!(
                    s.step_rhs, s.residuals,
                    s.variables, s.cones,
                    s.step_lhs, σ, μ
                )
            end

            @timeit s.timers "kkt solve" begin
            is_kkt_solve_success &= 
                kkt_solve!(
                    s.kktsystem, s.step_lhs, s.step_rhs,
                    s.data, s.variables, s.cones, :combined
                )
            end

            # We change scaling strategy on numerical error.  We 
            # take a small chance that the combined step will 
            # fail unchecked, and only put this logic here 
            if !is_kkt_solve_success
                # save scalars indicating no step 
                info_save_scalars(s.info,μ,zero(T),one(T),iter)
                if scaling_strategy == PrimalDual::ScalingStrategy
                    # switch to the more conservative dual scaling strategy 
                    # PJG: `continue`` means that we will compute residuals and  
                    # termination conditions twice for this iterate 
                    scaling_strategy = Dual::ScalingStrategy
                    println("Numerics: Switching to Dual strategy")
                    continue 
                elseif scaling_strategy == Dual::ScalingStrategy 
                    #out of tricks.  Bail out with an error 
                    s.info.status = NUMERICAL_ERROR
                    println("Break : Numerical Error")
                    break
                end
            end

            #compute final step length and update the current iterate
            #--------------
            α = calc_step_length(
                s.variables, s.step_lhs, s.work_vars,
                s.cones,s.settings,:combined, scaling_strategy
            )  

            α *= s.settings.max_step_fraction

            # YC: check if the step size is too small
            if scaling_strategy == PrimalDual::ScalingStrategy && 
                α < s.settings.min_primaldual_step_length
                   scaling_strategy = Dual
                   println("Progress: Switching to dual scaling")
                   #PJG: We are taking this final step anyway... 

            elseif scaling_strategy == Dual::ScalingStrategy && 
                α < s.settings.min_dual_step_length
                    s.info.status = INSUFFICIENT_PROGRESS
                    # save scalars indicating no step 
                    info_save_scalars(s.info,μ,zero(T),one(T),iter)
                    break
            end
            
            # Copy previous iterate in case the next one is a dud
            info_save_prev_iterate(s.info,s.variables,s.work_vars)

            variables_add_step!(s.variables,s.step_lhs,α)

            #record scalar values from this iteration
            info_save_scalars(s.info,μ,α,σ,iter)

        end  #end while
        #----------
        #----------

        end #end IP iteration timer

    end #end solve! timer

    info_finalize!(s.info,s.timers)  #halts timers
    solution_finalize!(s.solution,s.data,s.variables,s.info,s.settings)

    @notimeit info_print_footer(s.info,s.settings)

    return s.solution
end


# Mehrotra heuristic
function calc_centering_parameter(α::T) where{T}

    return σ = (1-α)^3
end


function solver_default_start!(s::Solver{T}) where {T}
    # YC:If there are only smmetric cones, use Mehrotra initialization strategy as ECOS and CVXOPT
    # Otherwise, initialize it along central rays
    if (s.cones.sym_flag)
        #set all scalings to identity (or zero for the zero cone)
        cones_set_identity_scaling!(s.cones)
        #Refactor
        kkt_update!(s.kktsystem,s.data,s.cones)
        #solve for primal/dual initial points via KKT
        kkt_solve_initial_point!(s.kktsystem,s.variables,s.data)
        #fix up (z,s) so that they are in the cone
        variables_shift_to_cone!(s.variables, s.cones)
    else
        #Unit initialization when there are asymmetric cones
        asymmetric_init_cone!(s.variables, s.cones)
    end

    return nothing
end

function Base.show(io::IO, solver::Clarabel.Solver{T}) where {T}
    println(io, "Clarabel model with Float precision: $(T)")
end
