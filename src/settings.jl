# -------------------------------------
# User definable settings
# -------------------------------------
Base.@kwdef mutable struct Settings{T <: AbstractFloat}

    max_iter::DefaultInt    = 50
    time_limit::T           = 0.   #unbounded if = 0
    verbose::Bool           = true
    tol_gap_abs::T          = 1e-8
    tol_gap_rel::T          = 1e-8
    tol_feas::T             = 1e-5
    direct_kkt_solver::Bool = true

    max_step_fraction::T    = 0.99

    #static regularization parameters
    static_regularization_enable::Bool  = true
    static_regularization_eps::T        = 1e-8

    #dynamic regularization parameters
    dynamic_regularization_enable::Bool = true
    dynamic_regularization_eps::T       = 1e-13
    dynamic_regularization_delta::T     = 2e-7

    #the direct linear solver to use
    #can be :qdldl or :mkl
    direct_solve_method::Symbol         = :qdldl

    #iterative refinement (for QDLDL)
    iterative_refinement_enable::Bool   = true
    iterative_refinement_reltol::T      = 1e-10
    iterative_refinement_abstol::T      = 1e-10
    iterative_refinement_max_iter::Int  = 10
    iterative_refinement_stop_ratio::T  = 2.

    #data equilibration
    equilibrate_enable::Integer         = true
    equilibrate_max_iter::Integer       = 10
    equilibrate_min_scaling::T          = 1e-4
    equilibrate_max_scaling::T          = 1e+4

end

Settings(args...) = Settings{DefaultFloat}(args...)


function Settings(d::Dict)

	settings = Settings()
	settings_populate!(d)
    return settings
end


function settings_populate!(settings::Settings, d::Dict)
    for (key, val) in d
        setfield!(settings, Symbol(key), val)
    end
    return nothing
end
