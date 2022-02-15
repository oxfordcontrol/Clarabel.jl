# -------------------------------------
# User definable settings
# -------------------------------------
mutable struct Settings{T <: AbstractFloat}

    max_iter::DefaultInt
    verbose::Bool
    tol_gap_abs::T
    tol_gap_rel::T
    tol_feas::T
    direct_kkt_solver::Bool

    max_step_fraction::T

    #static regularization parameters
    static_regularization_enable::Bool
    static_regularization_eps::T

    #dynamic regularization parameters
    dynamic_regularization_enable::Bool
    dynamic_regularization_eps::T
    dynamic_regularization_delta::T

    #iterative refinement
    iterative_refinement_enable::Bool
    iterative_refinement_reltol::T
    iterative_refinement_abstol::T
    iterative_refinement_max_iter::Int
    iterative_refinement_halt_ratio::T

    function Settings{T}(;

        max_iter = 50,
        verbose = true,
        tol_gap_abs = 1e-7,
        tol_gap_rel = 1e-6,
        tol_feas    = 1e-5,
        direct_kkt_solver = true,
        max_step_fraction = 0.99,
        #
        static_regularization_enable = true,
        static_regularization_eps = 1e-8,
        #
        dynamic_regularization_enable = true,
        dynamic_regularization_eps = 1e-12,
        dynamic_regularization_delta = 1e-7,
        #
        iterative_refinement_enable = true,
        iterative_refinement_reltol = 1e-10,
        iterative_refinement_abstol = 1e-10,
        iterative_refinement_max_iter = 10,
        iterative_refinement_halt_ratio = 2.) where {T}

        new(max_iter,verbose,
        tol_gap_abs,tol_gap_rel,tol_feas,
        direct_kkt_solver,max_step_fraction,
        static_regularization_enable,
        static_regularization_eps,
        dynamic_regularization_enable,
        dynamic_regularization_eps,
        dynamic_regularization_delta,
        iterative_refinement_enable,
        iterative_refinement_reltol,
        iterative_refinement_abstol,
        iterative_refinement_max_iter,
        iterative_refinement_halt_ratio
        )

    end
end

# Default to DefaultFloat type for reals
Settings(args...; kwargs...) = Settings{DefaultFloat}(args...; kwargs...)
