using PrettyTables

# -------------------------------------
# User definable settings
# -------------------------------------

"""
Argument | Default Value | Description
:--- | :--- | :---
||
__Main Algorithm Settings__||
||
max\\_iter                              | 50         | maximum number of iterations
time\\_limit                            | Inf        | maximum run time (seconds)
verbose                                 | true       | verbose printing
max\\_step\\_fraction                   | 0.99       | maximum interior point step length
||
__Full Accuracy Settings__||
tol\\_gap\\_abs                         | 1e-8       | absolute duality gap tolerance
tol\\_gap\\_rel                         | 1e-8       | relative duality gap tolerance
tol\\_feas                              | 1e-8       | feasibility check tolerance (primal and dual)
tol\\_infeas\\_abs						| 1e-8		 | absolute infeasibility tolerance (primal and dual)
tol\\_infeas\\_rel						| 1e-8		 | relative infeasibility tolerance (primal and dual)
tol\\_ktratio                           | 1e-7       | κ/τ tolerance
||
__Reduced Accuracy Settings__||
reduced\\_tol\\_gap\\_abs               | 5e-5       | reduced absolute duality gap tolerance
reduced\\_tol\\_gap\\_rel               | 5e-5       | reduced relative duality gap tolerance
reduced\\_tol\\_feas                    | 1e-4       | reduced feasibility check tolerance (primal and dual)
reduced\\_tol\\_infeas_abs		        | 5e-5       | reduced absolute infeasibility tolerance (primal and dual)
reduced\\_tol\\_infeas_rel		        | 5e-5       | reduced relative infeasibility tolerance (primal and dual)
reduced\\_tol\\_ktratio                 | 1e-4       | reduced κ/τ tolerance
||
__Data Equilibration Settings__||
||
equilibrate\\_enable                    | true       | enable data equilibration pre-scaling
equilibrate\\_max\\_iter                | 10         | maximum equilibration scaling iterations
equilibrate\\_min\\_scaling             | 1e-4       | minimum equilibration scaling allowed
equilibrate\\_max\\_scaling             | 1e+4       | maximum equilibration scaling allowed
||
__Step Size Settings__||
linesearch\\_backtrack\\_step           | 0.8        | linesearch backtracking
min\\_switch\\_step\\_length            | 1e-1       | minimum step size allowed for asymmetric cones with PrimalDual scaling
min\\_terminate\\_step\\_length         | 1e-4       | minimum step size allowed for symmetric cones && asymmetric cones with Dual scaling
||
__Linear Solver Settings__||
||
kkt\\_solver_method                     | :directldl | KKT solver method 
direct\\_solve\\_method                 | :qdldl     | direct LDL linear solver (:qdldl, :mkl or :cholmod)
static\\_regularization\\_enable        | true       | enable KKT static regularization
static\\_regularization\\_eps           | 1e-7       | KKT static regularization parameter
static\\_regularization\\_proportional  | eps(T)^2   | additional regularization parameter w.r.t. the maximum abs diagonal term
dynamic\\_regularization\\_enable       | true       | enable KKT dynamic regularization
dynamic\\_regularization\\_eps          | 1e-13      | KKT dynamic regularization threshold
dynamic\\_regularization\\_delta        | 2e-7       | KKT dynamic regularization shift
iterative\\_refinement\\_enable         | true       | KKT solve with iterative refinement
iterative\\_refinement\\_reltol         | 1e-12      | iterative refinement relative tolerance
iterative\\_refinement\\_abstol         | 1e-12      | iterative refinement absolute tolerance
iterative\\_refinement\\_max\\_iter     | 10         | iterative refinement maximum iterations
iterative\\_refinement\\_stop\\_ratio   | 5.0        | iterative refinement stalling tolerance
__Preprocessing Settings 
presolve_enable                         | true       | enable presolve constraint reduction

"""
Base.@kwdef mutable struct Settings{T <: AbstractFloat}

    max_iter::UInt32    	= 200
    time_limit::Float64     = Inf
    verbose::Bool           = true
    max_step_fraction::T    = 0.99

    # full accuracy solution tolerances
    tol_gap_abs::T          = 1e-8
    tol_gap_rel::T          = 1e-8
    tol_feas::T             = 1e-8
	tol_infeas_abs::T		= 1e-8
	tol_infeas_rel::T		= 1e-8
    tol_ktratio::T          = 1e-6

    # reduced accuracy solution tolerances
    reduced_tol_gap_abs::T          = 5e-5
    reduced_tol_gap_rel::T          = 5e-5
    reduced_tol_feas::T             = 1e-4
    reduced_tol_infeas_abs::T		= 5e-5
	reduced_tol_infeas_rel::T		= 5e-5
    reduced_tol_ktratio::T          = 1e-4

	#data equilibration
	equilibrate_enable::Bool            = true
	equilibrate_max_iter::UInt32        = 10
	equilibrate_min_scaling::T          = 1e-4
	equilibrate_max_scaling::T          = 1e+4

    #cones and line search parameters
    linesearch_backtrack_step::T        = 0.8     
    min_switch_step_length::T           = 1e-1   
    min_terminate_step_length::T        = 1e-4    

    #the direct linear solver to use
    #can be :qdldl or :mkl
    kkt_solver_method::Symbol           = :directldl  
    direct_solve_method::Symbol         = :qdldl

    #static regularization parameters
    static_regularization_enable::Bool    = true
    static_regularization_constant::T     = 1e-8     
    static_regularization_proportional::T = eps()^2 

    #dynamic regularization parameters
    dynamic_regularization_enable::Bool = true
    dynamic_regularization_eps::T       = 1e-13
    dynamic_regularization_delta::T     = 2e-7

    #iterative refinement
    iterative_refinement_enable::Bool   = true
    iterative_refinement_reltol::T      = 1e-13      
    iterative_refinement_abstol::T      = 1e-12 

    iterative_refinement_max_iter::Int  = 10
    iterative_refinement_stop_ratio::T  = 5     
    
    #preprocessing 
    presolve_enable::Bool               = true

end

Settings(args...) = Settings{DefaultFloat}(args...)

function Settings{T}(d::Dict) where{T}

	settings = Settings{T}()
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
