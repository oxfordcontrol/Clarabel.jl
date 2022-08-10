using PrettyTables

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
time_limit                              | Inf       | maximum run time (seconds)
verbose                                 | true      | verbose printing
tol\\_gap\\_abs                         | 1e-8      | absolute residual tolerance
tol\\_gap\\_rel                         | 1e-8      | relative residual tolerance
tol\\_feas                              | 1e-5      | feasibility check tolerance
tol\\_infeas\\_abs						| 1e-8		| absolute infeasibility tolerance
tol\\_infeas\\_rel						| 1e-8		| relative infeasibility tolerance
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
    time_limit::T           = Inf
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
    static_regularization_enable::Bool    = true
    static_regularization_constant::T     = 1e-8   #PJG: Add to docs
    static_regularization_proportional::T = eps(T) #PJG: Add to docs

    #dynamic regularization parameters
    dynamic_regularization_enable::Bool = true
    dynamic_regularization_eps::T       = 1e-13
    dynamic_regularization_delta::T     = 2e-7

    #iterative refinement
    iterative_refinement_enable::Bool   = true
    iterative_refinement_reltol::T      = 1e-10
    iterative_refinement_abstol::T      = 1e-10
    iterative_refinement_max_iter::Int  = 10
    iterative_refinement_stop_ratio::T  = 2

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


function Base.show(io::IO, settings::Clarabel.Settings{T}) where {T}

    s = get_precision_string(T)
    println("Clarabel settings with Float precision: $s\n")

	names   = fieldnames(Clarabel.Settings)
	valstrs = []
	types   = []
    for name in names
        value  = getfield(settings,name)
		type = typeof(value)
        push!(types, type)
        push!(valstrs,type == BigFloat ? @sprintf("%g",value) : string(value))
    end
	table = hcat(collect(names), types, valstrs)

	#NB: same as tf_compact, but with bolded row separator
	tf = TextFormat(
    up_right_corner     = ' ',
    up_left_corner      = ' ',
    bottom_left_corner  = ' ',
    bottom_right_corner = ' ',
    up_intersection     = ' ',
    left_intersection   = ' ',
    right_intersection  = ' ',
    middle_intersection = ' ',
    bottom_intersection  = ' ',
    column              = ' ',
    row                 = '='
   )

	header = ["Setting", "DataType", "Value"]

    pretty_table(table,
		header=header,
		compact_printing=true,
		alignment = :l,
		backend = Val(:text),
		tf = tf,
		hlines = [1])

end
