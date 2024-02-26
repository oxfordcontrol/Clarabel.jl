# -------------------------------------
# User definable settings
# -------------------------------------

"""
Argument | Default Value | Description
:--- | :--- | :---
||
__Main Algorithm Settings__||
||
max\\_iter                              | 50        | maximum number of iterations
time\\_limit                            | Inf       | maximum run time (seconds)
verbose                                 | true      | verbose printing
max\\_step\\_fraction                   | 0.99      | maximum interior point step length
||
__Full Accuracy Settings__||
tol\\_gap\\_abs                         | 1e-8      | absolute duality gap tolerance
tol\\_gap\\_rel                         | 1e-8      | relative duality gap tolerance
tol\\_feas                              | 1e-8      | feasibility check tolerance (primal and dual)
tol\\_infeas\\_abs						| 1e-8		| absolute infeasibility tolerance (primal and dual)
tol\\_infeas\\_rel						| 1e-8		| relative infeasibility tolerance (primal and dual)
tol\\_ktratio                           | 1e-7      | κ/τ tolerance
||
__Reduced Accuracy Settings__||
reduced\\_tol\\_gap\\_abs               | 5e-5      | reduced absolute duality gap tolerance
reduced\\_tol\\_gap\\_rel               | 5e-5      | reduced relative duality gap tolerance
reduced\\_tol\\_feas                    | 1e-4      | reduced feasibility check tolerance (primal and dual)
reduced\\_tol\\_infeas_abs		        | 5e-5      | reduced absolute infeasibility tolerance (primal and dual)
reduced\\_tol\\_infeas_rel		        | 5e-5      | reduced relative infeasibility tolerance (primal and dual)
reduced\\_tol\\_ktratio                 | 1e-4      | reduced κ/τ tolerance
||
__Data Equilibration Settings__||
||
equilibrate\\_enable                    | true      | enable data equilibration pre-scaling
equilibrate\\_max\\_iter                | 10        | maximum equilibration scaling iterations
equilibrate\\_min\\_scaling             | 1e-5      | minimum equilibration scaling allowed
equilibrate\\_max\\_scaling             | 1e+5      | maximum equilibration scaling allowed
||
__Step Size Settings__||
linesearch\\_backtrack\\_step           | 0.8       | linesearch backtracking
min\\_switch\\_step\\_length            | 1e-1      | minimum step size allowed for asymmetric cones with PrimalDual scaling
min\\_terminate\\_step\\_length         | 1e-4      | minimum step size allowed for symmetric cones && asymmetric cones with Dual scaling
||
__Linear Solver Settings__||
||
direct\\_kkt\\_solver                   | true      | use a direct linear solver method (required true)
direct\\_solve\\_method                 | :qdldl    | direct linear solver (:qdldl, :mkl or :cholmod)
static\\_regularization\\_enable        | true      | enable KKT static regularization
static\\_regularization\\_eps           | 1e-7      | KKT static regularization parameter
static\\_regularization\\_proportional  | eps(T)^2  | additional regularization parameter w.r.t. the maximum abs diagonal term
dynamic\\_regularization\\_enable       | true      | enable KKT dynamic regularization
dynamic\\_regularization\\_eps          | 1e-13     | KKT dynamic regularization threshold
dynamic\\_regularization\\_delta        | 2e-7      | KKT dynamic regularization shift
iterative\\_refinement\\_enable         | true      | KKT solve with iterative refinement
iterative\\_refinement\\_reltol         | 1e-12     | iterative refinement relative tolerance
iterative\\_refinement\\_abstol         | 1e-12     | iterative refinement absolute tolerance
iterative\\_refinement\\_max\\_iter     | 10        | iterative refinement maximum iterations
iterative\\_refinement\\_stop\\_ratio   | 5.0       | iterative refinement stalling tolerance
__Preprocessing Settings 
presolve_enable                         | true      | enable presolve constraint reduction

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
	equilibrate_min_scaling::T          = 1e-5
	equilibrate_max_scaling::T          = 1e+5

    #cones and line search parameters
    linesearch_backtrack_step::T        = 0.8     
    min_switch_step_length::T           = 1e-1   
    min_terminate_step_length::T        = 1e-4    

    #the direct linear solver to use
    #can be :qdldl or :mkl
    direct_kkt_solver::Bool             = true   #indirect not yet supported
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
    println(io, "Clarabel settings with Float precision: $s\n")

	names   = fieldnames(Clarabel.Settings)
	values  = String[]
	types   = String[]
    for name in names
        v = getfield(settings,name)
		type = String(Symbol(typeof(v)))
        push!(types, type)
        push!(values,type == BigFloat ? @sprintf("%g",v) : string(v))
    end
    names = collect(String.(names))
    types = String.(Symbol.(types))

    table = [names types values]
    titles = ["Setting","DataType","Value"]

    # pad out each column of the table to a common length 
    dividers = String[]
    for i in eachindex(titles)
        len = max(8,maximum(length.(table[:,i])))
        table[:,i] .= rpad.(table[:,i],len+1)
        titles[i]  = rpad(titles[i],len+1)
        push!(dividers,repeat("=",len+2))
    end

    # print the header ...
    @printf(io, " ")
    for str in titles
        @printf(io, " %s ", str)
    end 
    println(io)
    # and the divider ...
    @printf(io, " ")
    for str in dividers
        @printf(io, "%s ", str)  
    end 
    println(io)
    # and the settings 
    for row in 1:size(table,1)
        @printf(io, " ")
        for col in 1:size(table,2)
            @printf(io, " %s ", table[row,col])
        end
        println(io)
    end
    println(io)

end
