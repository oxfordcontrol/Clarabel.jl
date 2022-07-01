using DataFrames

# -------------------------------------
# User definable settings
# -------------------------------------

"""
	Clarabel.Settings{T}(kwargs) where {T <: AbstractFloat}

Creates a Clarabel Settings object that is used to pass user settings to the solver.

Argument | Default Value | Description
:--- | :--- | :---
||
__Main Algorithm Settings__||
||
max_iter                                | 50        | maximum number of iterations
time_limit                              | 0         | maximum run time (seconds)
verbose                                 | true      | verbose printing
tol\\_gap\\_abs                         | 1e-8      | absolute residual tolerance
tol\\_gap\\_rel                         | 1e-8      | relative residual tolerance
tol\\_feas                              | 1e-5      | feasibility check tolerance
tol_\\infeas\\_abs						| 1e-8		| absolute infeasibility tolerance
tol_\\infeas\\_rel						| 1e-8		| relative infeasibility tolerance
max\\_step\\_fraction                   | 0.99      | maximum interior point step length
||
__Data Equilibration Settings__||
||
equilibrate\\_enable                    | true      | enable  data equilibration pre-scaling
equilibrate\\_max\\_iter                | 10        | maximum equilibration scaling iterations
equilibrate\\_min\\_scaling             | 1e-4      | minimum equilibration scaling allowed
equilibrate\\_max\\_scaling             | 1e+4      | maximum equilibration scaling allowed
||
	__Linear Solver Settings__||
||
direct\\_kkt\\_solver                   | true      | use a direct linear solver method (required true)
direct\\_solve\\_method                 | :qdldl    | direct linear solver (:qdldl, :mkl or :cholmod)
static\\_regularization\\_enable        | true      | enable KKT static regularization
static\\_regularization\\_eps           | 1e-8      | KKT static regularization parameter
dynamic\\_regularization\\_enable       | true      | enable KKT dynamic regularization
dynamic\\_regularization\\_eps          | 1e-13     | KKT dynamic regularization threshold
dynamic\\_regularization\\_delta        | 2e-7      | KKT dynamic regularization shift
iterative\\_refinement\\_enable         | true      | KKT solve with iterative refinement
iterative\\_refinement\\_reltol         | 1e-10     | iterative refinement relative tolerance
iterative\\_refinement\\_abstol         | 1e-10     | iterative refinement absolute tolerance
iterative\\_refinement\\_max\\_iter     | 10        | iterative refinement maximum iterations
iterative\\_refinement\\_stop\\_ratio   | 2.0       | iterative refinement stalling tolerance

"""
Base.@kwdef mutable struct Settings{T <: AbstractFloat}

    max_iter::DefaultInt    = 50
    time_limit::T           = 0.   #unbounded if = 0
    verbose::Bool           = true
    tol_gap_abs::T          = 1e-8
    tol_gap_rel::T          = 1e-8
    tol_feas::T             = 1e-5
	tol_infeas_abs::T		= 1e-8
	tol_infeas_rel::T		= 1e-8
    max_step_fraction::T    = 0.99

	#data equilibration
	equilibrate_enable::Bool            = true
	equilibrate_max_iter::Integer       = 10
	equilibrate_min_scaling::T          = 1e-4
	equilibrate_max_scaling::T          = 1e+4

    #the direct linear solver to use
    #can be :qdldl or :mkl
    direct_kkt_solver::Bool             = true   #indirect not yet supported
    direct_solve_method::Symbol         = :qdldl

    #static regularization parameters
    static_regularization_enable::Bool  = true
    static_regularization_eps::T        = 1e-8

    #dynamic regularization parameters
    dynamic_regularization_enable::Bool = true
    dynamic_regularization_eps::T       = 1e-13
    dynamic_regularization_delta::T     = 2e-7

    # proportional regularization parameter as in mosek
    proportional_eps = T(0)

    #iterative refinement (for QDLDL)
    iterative_refinement_enable::Bool   = true
    iterative_refinement_reltol::T      = 1e-10
    iterative_refinement_abstol::T      = 1e-10
    iterative_refinement_max_iter::Int  = 10
    iterative_refinement_stop_ratio::T  = 1.01

end

Settings(args...) = Settings{DefaultFloat}(args...)


function Settings(d::Dict)

	settings = Settings()
	settings_populate!(settings,d)
    return settings
end


function settings_populate!(settings::Settings, d::Dict)
    for (key, val) in d
        setfield!(settings, Symbol(key), val)
    end
    return nothing
end


# Inspired by method in https://discourse.julialang.org/t/how-to-align-output-in-columns/3938/2

function Base.show(io::IO, settings::Clarabel.Settings{T}) where {T}

    s = get_precision_string(T)
    println("Clarabel settings with Float precision: $s\n")

    df = DataFrame(Setting = Symbol[], DataType = DataType[], Value = String[])

    for name in fieldnames(Clarabel.Settings)
        value  = getfield(settings,name)
        type   = typeof(value)
        valstr = type == BigFloat ? @sprintf("%g",value) : string(value)
        push!(df, [name, type, valstr])
    end

    strwidths = [maximum(textwidth.(string.([df[:, i]; names(df)[i]]))) for i in 1:size(df, 2)]
    io = IOBuffer()

    # Print headers
    for (i, header) in enumerate(names(df))
        print(io, rpad(header, strwidths[i]), "   ")
    end
    println(io)

    # Print separator
    for (i, header) in enumerate(names(df))
        print(io, "="^strwidths[i], "   ")
    end
    println(io)

    for j in 1:size(df, 1)
        for i in 1:size(df, 2)
            print(io, rpad(df[j,i], strwidths[i]), "   ")
        end
        println(io)
    end

    print(String(take!(io)))
end
