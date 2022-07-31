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
    α::Vector{Union{Nothing, T}};
    kwargs...
) where{T <: AbstractFloat}

    s = Solver{T}()
    setup!(s,P,c,A,b,cone_types,cone_dims,α,kwargs...)
    return s
end

# -------------------------------------
# setup!
# -------------------------------------


"""
	setup!(solver, P, q, A, b, cone_types, cone_dims, α, [settings])

Populates a [`Solver`](@ref) with a cost function defined by `P` and `q`, and one or more conic constraints defined by `A`, `b` and a description of a conic constraint composed of cones whose types and dimensions are in `cone_types` and `cone_dims`, respectively.
α is set to the exponent for a power cone and is set 'nothing' otherwise

The solver will be configured to solve the following optimization problem:

```
min   1/2 x'Px + q'x
s.t.  Ax + s = b, s ∈ K
```

All data matrices must be sparse.   The matrix `P` is assumed to be symmetric and positive semidefinite, and only the upper triangular part is used.

The cone `K` is a composite cone whose consituent cones are described by
* cone_types::Vector{Clarabel.SupportedCones}
* cone_dims::Vector{Int}
* α::Vector::Vector{Union{Nothing, T}}

The optional argument `settings` can be used to pass custom solver settings:
```julia
settings = Clarabel.Settings(verbose = true)
setup!(model, P, q, A, b, cone_types, cone_dims, α, settings)
```

To solve the problem, you must make a subsequent call to [`solve!`](@ref)
"""
function setup!(
    s::Solver{T},
    P::AbstractMatrix{T},
    q::Vector{T},
    A::AbstractMatrix{T},
    b::Vector{T},
    cone_types::Vector{SupportedCones},
    cone_dims::Vector{Int},
    α::Vector{Union{Nothing, T}} = Union{Nothing, T}[],
    settings::Settings{T} = Settings,
) where {T}
    #this allows total override of settings during setup
    s.settings = settings
    setup!(s,P,q,A,b,cone_types,cone_dims,α)
end

function setup!(
    s::Solver{T},
    P::AbstractMatrix{T},
    q::Vector{T},
    A::AbstractMatrix{T},
    b::Vector{T},
    cone_types::Vector{SupportedCones},
    cone_dims::Vector{Int},
    α::Vector{Union{Nothing, T}};
    kwargs...
) where {T}
    #this allows override of individual settings during setup
    settings_populate!(s.settings, Dict(kwargs))
    setup!(s,P,q,A,b,cone_types,cone_dims,α)
end

# main setup function
function setup!(
    s::Solver{T},
    P::AbstractMatrix{T},
    q::Vector{T},
    A::AbstractMatrix{T},
    b::Vector{T},
    cone_types::Vector{SupportedCones},
    cone_dims::Vector{Int},
    α::Vector{Union{Nothing, T}},
) where{T}

    #make this first to create the timers
    s.info    = DefaultInfo{T}()

    @timeit s.info.timer "setup!" begin

        s.cones  = ConeSet{T}(cone_types,cone_dims,α)
        s.data   = DefaultProblemData{T}(P,q,A,b,s.cones)
        s.data.m == s.cones.numel || throw(DimensionMismatch())

        s.variables = DefaultVariables{T}(s.data.n,s.cones)
        s.residuals = DefaultResiduals{T}(s.data.n,s.data.m)

        #equilibrate problem data immediately on setup.
        #this prevents multiple equlibrations if solve!
        #is called more than once.
        @timeit_debug s.info.timer "equilibrate" begin
            data_equilibrate!(s.data,s.cones,s.settings)
        end

        @timeit_debug s.info.timer "kkt init" begin
            s.kktsystem = DefaultKKTSystem{T}(s.data,s.cones,s.settings)
        end

        # work variables for assembling step direction LHS/RHS
        s.step_rhs  = DefaultVariables{T}(s.data.n,s.cones)
        s.step_lhs  = DefaultVariables{T}(s.data.n,s.cones)
        s.workVar   = DefaultVariables{T}(s.data.n,s.cones)

        # user facing results go here
        s.result    = Result{T}(s.data.m,s.data.n,s.info.timer)

    end

    return s
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
    info_reset!(s.info)
    iter   = 0
    isdone = false
    timer  = s.info.timer

    #initial residuals and duality gap
    gap       = T(0)
    sigma     = T(0)

    #solver release info, solver config
    #problem dimensions, cone type etc
    @notimeit print_header(s.info,s.settings,s.data,s.cones)

    #NB: temporary allocation
    #PJG: This should be removed.
    # workVar = DefaultVariables{T}(s.data.n,s.cones)

    @timeit timer "solve!" begin

        #initialize variables to some reasonable starting point
        #@timeit_debug timer "default start" 
        solver_default_start!(s)

        @timeit_debug timer "IP iteration" begin

        #----------
        # main loop
        #----------

        while true
            #update the residuals
            #--------------
            @timeit_debug timer "residuals_update" residuals_update!(s.residuals,s.variables,s.data)

            #calculate duality gap (scaled)
            #--------------
            μ = calc_mu(s.variables, s.residuals, s.cones)

            #convergence check and printing
            #--------------
            #@timeit_debug timer "check termination" 
            begin
                info_update!(
                    s.info,s.data,s.variables,
                    s.residuals,s.settings
                )
                isdone = info_check_termination!(s.info,s.residuals,s.settings)
            end

            iter += 1
            @notimeit print_status(s.info,s.settings)
            isdone && break

            #update the scalings
            #--------------
            @timeit_debug timer "NT scaling" scaling_update!(s.cones,s.variables,μ,s.kktsystem.kktsolver.corFlag)

            #update the KKT system and the constant
            #parts of its solution
            #--------------
            #@timeit_debug timer "kkt update" 
            kkt_update!(s.kktsystem,s.data,s.cones)

            #calculate the affine step
            #--------------
            @timeit_debug timer "calc_affine_step_rhs" begin
                calc_affine_step_rhs!(
                    s.step_rhs, s.residuals,
                    s.variables, s.cones
                )
            end

            @timeit_debug timer "kkt solve affine" begin
                kkt_solve!(
                    s.kktsystem, s.step_lhs, s.step_rhs,
                    s.data, s.variables, s.cones, :affine
                )
            end

            # check_KKT_system!(
            #     s.kktsystem, s.step_lhs, s.step_rhs,
            #     s.data, s.variables, s.cones)

            #calculate step length and centering parameter
            #--------------
            @timeit_debug timer "step length affine" begin
                α = calc_step_length(s.variables,s.step_lhs,s.workVar,s.cones,:affine)
                σ = calc_centering_parameter(α)
            end
            # println("σ is: ", σ)

            #calculate the combined step and length
            #--------------
            @timeit_debug timer "calc_combined_step_rhs" begin
                calc_combined_step_rhs!(
                    s.step_rhs, s.residuals,
                    s.variables, s.cones,
                    s.step_lhs, σ, μ
                )
            end

            @timeit_debug timer "kkt solve combined" begin
                kkt_solve!(
                    s.kktsystem, s.step_lhs, s.step_rhs,
                    s.data, s.variables, s.cones, :combined
                )
            end

            #compute final step length and update the current iterate
            #--------------
            @timeit_debug timer "step length" begin
                α = calc_step_length(s.variables,s.step_lhs,s.workVar,s.cones,:combined)
            end

            while (α < 0.1 && σ < one(T))
                println("step size too small!! with σ is ", σ)
                σ *= 10
                calc_combined_step_rhs!(
                    s.step_rhs, s.residuals,
                    s.variables, s.cones,
                    s.step_lhs, σ, μ
                )
                kkt_solve!(
                    s.kktsystem, s.step_lhs, s.step_rhs,
                    s.data, s.variables, s.cones, :combined
                )
                α = calc_step_length(s.variables,s.step_lhs,s.workVar,s.cones,:combined)
                println("update α ", α)
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

            # #update the scalings
            # #--------------
            # @timeit_debug timer "NT scaling" scaling_update!(s.cones,s.variables,μ)

        end  #end while
        #----------
        #----------

        end #end IP iteration timer

    end #end solve! timer

    info_finalize!(s.info)  #halts timers
    result_finalize!(s.result,s.data,s.variables,s.info)

    @notimeit print_footer(s.info,s.settings)

    return s.result
end

# function solve!(
#     s::Solver{T}
# ) where{T}

#     #various initializations
#     info_reset!(s.info)
#     iter   = 0
#     isdone = false
#     timer  = s.info.timer

#     #initial residuals and duality gap
#     gap       = T(0)
#     sigma     = T(0)

#     #solver release info, solver config
#     #problem dimensions, cone type etc
#     @notimeit print_header(s.info,s.settings,s.data,s.cones)

#     #NB: temporary allocation
#     #PJG: This should be removed.
#     # workVar = DefaultVariables{T}(s.data.n,s.cones)

#     @timeit timer "solve!" begin

#         #initialize variables to some reasonable starting point
#         #@timeit_debug timer "default start" 
#         solver_default_start!(s)

#         @timeit_debug timer "IP iteration" begin

#         #----------
#         # main loop
#         #----------

#         while true
#             to = TimerOutput()
#             #update the residuals
#             #--------------
#             @timeit to  "residuals_update" residuals_update!(s.residuals,s.variables,s.data)

#             #calculate duality gap (scaled)
#             #--------------
#             μ = calc_mu(s.variables, s.residuals, s.cones)

#             #convergence check and printing
#             #--------------
#             @timeit to  "check termination" begin
#                 info_update!(
#                     s.info,s.data,s.variables,
#                     s.residuals,s.settings
#                 )
#                 isdone = info_check_termination!(s.info,s.residuals,s.settings)
#             end

#             iter += 1
#             @notimeit print_status(s.info,s.settings)
#             isdone && break

#             #update the scalings
#             #--------------
#             @timeit to "NT scaling" scaling_update!(s.cones,s.variables,μ,s.kktsystem.kktsolver.corFlag)

#             #update the KKT system and the constant
#             #parts of its solution
#             #--------------
#             #@timeit_debug timer "kkt update" 
#             @timeit to "kkt_update" kkt_update!(s.kktsystem,s.data,s.cones)

#             #calculate the affine step
#             #--------------
#             @timeit to "calc_affine_step_rhs" begin
#                 calc_affine_step_rhs!(
#                     s.step_rhs, s.residuals,
#                     s.variables, s.cones
#                 )
#             end

#             @timeit to "kkt solve affine" begin
#                 kkt_solve!(
#                     s.kktsystem, s.step_lhs, s.step_rhs,
#                     s.data, s.variables, s.cones, :affine
#                 )
#             end

#             # check_KKT_system!(
#             #     s.kktsystem, s.step_lhs, s.step_rhs,
#             #     s.data, s.variables, s.cones)

#             #calculate step length and centering parameter
#             #--------------
#             @timeit to "step length affine" begin
#                 α = calc_step_length(s.variables,s.step_lhs,s.workVar,s.cones,:affine)
#                 σ = calc_centering_parameter(α)
#             end
#             # println("σ is: ", σ)

#             #calculate the combined step and length
#             #--------------
#             @timeit to "calc_combined_step_rhs" begin
#                 calc_combined_step_rhs!(
#                     s.step_rhs, s.residuals,
#                     s.variables, s.cones,
#                     s.step_lhs, σ, μ
#                 )
#             end

#             @timeit to "kkt solve combined" begin
#                 kkt_solve!(
#                     s.kktsystem, s.step_lhs, s.step_rhs,
#                     s.data, s.variables, s.cones, :combined
#                 )
#             end

#             #compute final step length and update the current iterate
#             #--------------
#             @timeit to "step length" begin
#                 α = calc_step_length(s.variables,s.step_lhs,s.workVar,s.cones,:combined)
#             end

#             @timeit to "alpha scale " α *= s.settings.max_step_fraction

#             @timeit to "variables_add_step" begin
#                 variables_add_step!(s.variables,s.step_lhs,α)
#             end

#             #record scalar values from this iteration
#             @timeit to "save scalars" begin
#                 info_save_scalars(s.info,μ,α,σ,iter)
#             end

#             # YC:: offset_KKT_diag directly
#             @timeit to "offset diag" offset_KKT_diag(s.kktsystem.kktsolver)

#             print_timer(to)
#             reset_timer!()

#             # #update the scalings
#             # #--------------
#             # @timeit_debug timer "NT scaling" scaling_update!(s.cones,s.variables,μ)

#         end  #end while
#         #----------
#         #----------

#         end #end IP iteration timer

#     end #end solve! timer

#     info_finalize!(s.info)  #halts timers
#     result_finalize!(s.result,s.data,s.variables,s.info)

#     @notimeit print_footer(s.info,s.settings)

#     return s.result
# end


# Mehrotra heuristic
function calc_centering_parameter(α::T) where{T}

    return σ = (1-α)^3
end


function solver_default_start!(s::Solver{T}) where {T}
    # YC:If there are only smmetric cones, use Mehrotra initialization strategy as ECOS and CVXOPT
    # Otherwise, initialize it along central rays
    if (s.cones.symFlag)
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
        unsymmetricInit(s.variables, s.cones)
    end

    # YC:: offset_P_diag directly
    offset_P_diag(s.kktsystem.kktsolver)
    return nothing
end

function Base.show(io::IO, solver::Clarabel.Solver{T}) where {T}
    println(io, "Clarabel model with Float precision: $(T)")
end

# YC:need to be removed later
function check_KKT_system!(
    kktsystem::DefaultKKTSystem{T},
    lhs::DefaultVariables{T},
    rhs::DefaultVariables{T},
    data::DefaultProblemData{T},
    variables::DefaultVariables{T},
    cones::ConeSet{T},
) where {T}
    m,n = size(data.A)
    ξ = variables.x/variables.τ
    K = [data.P data.A' data.q; -data.A spzeros(T,m,m) data.b; -(2*data.P*ξ + data.q)' -data.b' dot(ξ,data.P,ξ)]
    v1 = [zeros(T,n,1); lhs.s.vec; lhs.κ]
    v2 = [lhs.x; lhs.z.vec; lhs.τ]
    v3 = [rhs.x; rhs.z.vec; rhs.τ]
    res = v1 - K*v2+v3

    # Q = [kktsystem.kktsolver.ldlsolver.KKTsym vcat(data.q, -data.b); -(2*data.Psym*ξ + data.q)' -data.b' (dot(ξ,data.Psym,ξ)+variables.κ/variables.τ)]
    # w1 = [lhs.x; lhs.z.vec; lhs.τ]
    # w2 = [rhs.x; kktsystem.work_conic.vec-rhs.z.vec; rhs.τ - rhs.κ/variables.τ]
    # res = Q*w1 - w2

    println("KKT residual is: ", norm(res,Inf))

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