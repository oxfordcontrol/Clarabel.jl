# -------------------------------------
# utility constructor that includes
# both object creation and setup
#--------------------------------------
function Solver(
    P::AbstractMatrix{T},
    c::Vector{T},
    A::AbstractMatrix{T},
    b::Vector{T},
    cone_types::Vector{<:SupportedCone},
    kwargs...
) where{T <: AbstractFloat}

    s = Solver{T}()
    setup!(s,P,c,A,b,cone_types,kwargs...)
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
function setup!(s,P,c,A,b,cone_types,settings::Settings)
    #this allows total override of settings during setup
    s.settings = settings
    setup!(s,P,c,A,b,cone_types)
end

function setup!(s,P,c,A,b,cone_types; kwargs...)
    #this allows override of individual settings during setup
    settings_populate!(s.settings, Dict(kwargs))
    setup!(s,P,c,A,b,cone_types)
end

# main setup function
function setup!(
    s::Solver{T},
    P::AbstractMatrix{T},
    q::Vector{T},
    A::AbstractMatrix{T},
    b::Vector{T},
    cone_types::Vector{<:SupportedCone},
) where{T}

    #sanity check problem dimensions
    _check_dimensions(P,q,A,b,cone_types)

    #make this first to create the timers
    s.info    = DefaultInfo{T}()

    @timeit s.timers "setup!" begin

        s.cones  = ConeSet{T}(cone_types)
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

function _check_dimensions(P,q,A,b,cone_types)

    n = length(q)
    m = length(b)
    p = sum(cone -> nvars(cone), cone_types; init = 0)

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

        # PJG: come back to this to ensure that timer sections
        # and tabs remain consistent with the Rust labels
        # YC: Why we have some parts that are not timed?

        while true
            #update the residuals
            #--------------
            @timeit_debug s.timers "residuals_update" residuals_update!(s.residuals,s.variables,s.data)

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
                isdone = info_check_termination!(s.info,s.residuals,s.settings)
            end

            iter += 1
            @notimeit info_print_status(s.info,s.settings)
            isdone && break

            #update the scalings
            #--------------

            # PJG: This not a good general structure, since the flag
            # being fed down to the cones here is coming from deep
            # within the kkt system itself.   This is not easily made
            # generic across solvers or problem domains.
            # YC: The flag is to determine when we switch from primal-dual scaling to the dual scaling depending on the conditioning number of the KKT matrix. 

            variables_scale_cones!(s.variables,s.cones,μ,s.kktsystem.kktsolver.scale_flag)

            #update the KKT system and the constant
            #parts of its solution
            #--------------
            @timeit s.timers "kkt update" kkt_update!(s.kktsystem,s.data,s.cones)

            #calculate the affine step
            #--------------
            calc_affine_step_rhs!(
                s.step_rhs, s.residuals,
                s.variables, s.cones
            )

            @timeit s.timers "kkt solve" begin
                kkt_solve!(
                    s.kktsystem, s.step_lhs, s.step_rhs,
                    s.data, s.variables, s.cones, :affine
                )
            end

            #calculate step length and centering parameter
            #--------------
            @timeit_debug timer "step length affine" begin
                α = calc_step_length(s.variables,s.step_lhs,s.work_vars,s.cones,:affine)
                σ = calc_centering_parameter(α)
            end

            #calculate the combined step and length
            #--------------
            calc_combined_step_rhs!(
                s.step_rhs, s.residuals,
                s.variables, s.cones,
                s.step_lhs, σ, μ
            )

            @timeit s.timers "kkt solve" begin
                kkt_solve!(
                    s.kktsystem, s.step_lhs, s.step_rhs,
                    s.data, s.variables, s.cones, :combined
                )
            end

            #compute final step length and update the current iterate
            #--------------
            @timeit_debug timer "step length final" begin
                α = calc_step_length(s.variables,s.step_lhs,s.work_vars,s.cones,:combined)
            end

            @timeit_debug timer "alpha scale " α *= s.settings.max_step_fraction

            @timeit_debug timer "variables_add_step" begin
                variables_add_step!(s.variables,s.step_lhs,α)
            end

            #record scalar values from this iteration
            @timeit_debug timer "save scalars" begin
                info_save_scalars(s.info,μ,α,σ,iter)
            end

            # YC:: offset_KKT_diag directly
            offset_KKT_diag(s.kktsystem.kktsolver)

        end  #end while
        #----------
        #----------

        end #end IP iteration timer

    end #end solve! timer

    info_finalize!(s.info,s.timers)  #halts timers
    solution_finalize!(s.solution,s.data,s.variables,s.info)

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
        #Unit initialization when there are unsymmetric cones
        unsymmetric_init!(s.variables, s.cones)
    end

    # YC:: offset_P_diag directly
    offset_P_diag(s.kktsystem.kktsolver)
    return nothing
end

function Base.show(io::IO, solver::Clarabel.Solver{T}) where {T}
    println(io, "Clarabel model with Float precision: $(T)")
end

# offset diagonal static regularization directly
function offset_P_diag(
    kktsolver::AbstractKKTSolver{T}
) where{T}
    settings  = kktsolver.settings
    map       = kktsolver.map
    ϵ = settings.static_regularization_eps
    KKT       = kktsolver.KKT

    # _offset_values!(kktsolver,map.diagP,-ϵ)  #undo the (now doubled) P shift

    if(settings.static_regularization_enable)
        (m,n,p) = (kktsolver.m,kktsolver.n,kktsolver.p)
        _offset_values!(kktsolver.ldlsolver,KKT, map.diag_full[1:n], -ϵ, kktsolver.Dsigns[1:n])
    end
end

function offset_KKT_diag(
    kktsolver::AbstractKKTSolver{T}
) where{T}
    settings  = kktsolver.settings
    map       = kktsolver.map
    ϵ = kktsolver.ϵ
    KKT       = kktsolver.KKT

    # _offset_values!(kktsolver,map.diagP,-ϵ)  #undo the (now doubled) P shift

    if(settings.static_regularization_enable)
        (m,n,p) = (kktsolver.m,kktsolver.n,kktsolver.p)
        _offset_values!(kktsolver.ldlsolver,KKT, map.diag_full, -ϵ, kktsolver.Dsigns)
    end
end
