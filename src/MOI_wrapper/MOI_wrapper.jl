# ======================================================================
#  Clarabel MOI wrapper (bucketed, preserves cone grouping order)
#  Grouping order: Zeros -> Nonnegatives -> SOC -> Exp -> Power -> PSD -> GenPower
# ======================================================================
using MathOptInterface, SparseArrays
using ..Clarabel
export Optimizer

#-----------------------------
# Const definitions
#-----------------------------

const MOI = MathOptInterface
const MOIU = MOI.Utilities
const SparseTriplet{T} = Tuple{Vector{Ti}, Vector{Ti}, Vector{T}} where {T,Ti <: Integer}
const DefaultInt = Clarabel.DefaultInt

# Cones supported by the solver
const OptimizerSupportedMOICones{T} = Union{
    MOI.Zeros,
    MOI.Nonnegatives,
    MOI.SecondOrderCone,
    MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle},
    MOI.ExponentialCone,
    MOI.PowerCone{T},
    Clarabel.MOI.GenPowerCone{T}
} where {T}

#Optimizer will consolidate cones of these types if possible
const OptimizerMergeableTypes = [Clarabel.ZeroConeT, Clarabel.NonnegativeConeT]
const OptimizerPriorGPUTypes = [MOI.Zeros, MOI.Nonnegatives]

#mappings between MOI and internal definitions

const MOItoClarabelCones = Dict([
    MOI.Zeros           => Clarabel.ZeroConeT,
    MOI.Nonnegatives    => Clarabel.NonnegativeConeT,
    MOI.SecondOrderCone => Clarabel.SecondOrderConeT,
    MOI.ExponentialCone => Clarabel.ExponentialConeT,
    MOI.PowerCone       => Clarabel.PowerConeT,
    Clarabel.MOI.GenPowerCone => Clarabel.GenPowerConeT,
    MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle} => Clarabel.PSDTriangleConeT,
])

# PJG: PrimalStatus/DualStatus just reported as "NEARLY_FEASIBLE"
# in the "ALMOST_SOLVED" cases.  We do not currently attempt 
# to distinguish cases that were only "almost" due to duality 
# gap / primal feasibility / dual feasibility.   The solver 
# convergence checks could be be written more finely to allow 
# separation of these different cases.  Note COSMO does 
# something along those lines.

const ClarabeltoMOITerminationStatus = Dict([
    Clarabel.SOLVED                     =>  MOI.OPTIMAL,
    Clarabel.MAX_ITERATIONS             =>  MOI.ITERATION_LIMIT,
    Clarabel.MAX_TIME                   =>  MOI.TIME_LIMIT,
    Clarabel.PRIMAL_INFEASIBLE          =>  MOI.INFEASIBLE,
    Clarabel.DUAL_INFEASIBLE            =>  MOI.DUAL_INFEASIBLE,
    Clarabel.ALMOST_SOLVED              =>  MOI.ALMOST_OPTIMAL,
    Clarabel.ALMOST_PRIMAL_INFEASIBLE   =>  MOI.ALMOST_INFEASIBLE,
    Clarabel.ALMOST_DUAL_INFEASIBLE     =>  MOI.ALMOST_DUAL_INFEASIBLE,
    Clarabel.NUMERICAL_ERROR            =>  MOI.NUMERICAL_ERROR,
    Clarabel.INSUFFICIENT_PROGRESS      =>  MOI.SLOW_PROGRESS,
])

const ClarabeltoMOIPrimalStatus = Dict([
    Clarabel.SOLVED                     =>  MOI.FEASIBLE_POINT,
    Clarabel.PRIMAL_INFEASIBLE          =>  MOI.INFEASIBLE_POINT,
    Clarabel.DUAL_INFEASIBLE            =>  MOI.INFEASIBILITY_CERTIFICATE,
    Clarabel.ALMOST_SOLVED              =>  MOI.NEARLY_FEASIBLE_POINT,
    Clarabel.ALMOST_PRIMAL_INFEASIBLE   =>  MOI.INFEASIBLE_POINT,
    Clarabel.ALMOST_DUAL_INFEASIBLE     =>  MOI.NEARLY_INFEASIBILITY_CERTIFICATE,
    Clarabel.MAX_ITERATIONS             =>  MOI.OTHER_RESULT_STATUS,
    Clarabel.MAX_TIME                   =>  MOI.OTHER_RESULT_STATUS,
    Clarabel.NUMERICAL_ERROR            =>  MOI.OTHER_RESULT_STATUS,
    Clarabel.INSUFFICIENT_PROGRESS      =>  MOI.OTHER_RESULT_STATUS,
])

const ClarabeltoMOIDualStatus = Dict([
    Clarabel.SOLVED                     =>  MOI.FEASIBLE_POINT,
    Clarabel.PRIMAL_INFEASIBLE          =>  MOI.INFEASIBILITY_CERTIFICATE,
    Clarabel.DUAL_INFEASIBLE            =>  MOI.INFEASIBLE_POINT,
    Clarabel.ALMOST_SOLVED              =>  MOI.NEARLY_FEASIBLE_POINT,
    Clarabel.ALMOST_PRIMAL_INFEASIBLE   =>  MOI.NEARLY_INFEASIBILITY_CERTIFICATE,
    Clarabel.ALMOST_DUAL_INFEASIBLE     =>  MOI.INFEASIBLE_POINT,
    Clarabel.MAX_ITERATIONS             =>  MOI.OTHER_RESULT_STATUS,
    Clarabel.MAX_TIME                   =>  MOI.OTHER_RESULT_STATUS,
    Clarabel.NUMERICAL_ERROR            =>  MOI.OTHER_RESULT_STATUS,
    Clarabel.INSUFFICIENT_PROGRESS      =>  MOI.OTHER_RESULT_STATUS,
])

#-----------------------------
# Main interface struct
#-----------------------------

mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    solver_module::Module
    solver::Union{Nothing,Clarabel.AbstractSolver{T}}
    solver_settings::Clarabel.Settings{T}
    solver_info::Union{Nothing,Clarabel.DefaultInfo{T}}
    solver_solution::Union{Nothing,Clarabel.DefaultSolution{T}}
    solver_nvars::Union{Nothing,DefaultInt}
    use_quad_obj::Bool
    sense::MOI.OptimizationSense
    objconstant::T
    # Map constraint indices to row ranges in the assembled A,b
    rowranges::Vector{UnitRange{DefaultInt}}

    function Optimizer{T}(; solver_module = Clarabel, user_settings...) where {T}
        solver_module   = solver_module
        solver          = nothing
        solver_settings = Clarabel.Settings{T}()
        solver_info     = nothing
        solver_solution = nothing
        solver_nvars    = nothing
        use_quad_obj    = true
        sense = MOI.MIN_SENSE
        objconstant = zero(T)
        rowranges = UnitRange{DefaultInt}[]

        optimizer = new(solver_module,solver,solver_settings,solver_info,solver_solution,solver_nvars,use_quad_obj,sense,objconstant,rowranges)
        for (key, value) in user_settings
            MOI.set(optimizer, MOI.RawOptimizerAttribute(string(key)), value)
        end
        return optimizer
    end
end

Optimizer(args...; kwargs...) = Optimizer{Clarabel.DefaultFloat}(args...; kwargs...)


#-----------------------------
# Required basic methods
#-----------------------------

# reset the optimizer
function MOI.empty!(optimizer::Optimizer{T}) where {T}
    #flush everything, keeping the currently configured settings
    optimizer.solver          = nothing
    optimizer.solver_settings = optimizer.solver_settings #preserve settings / no change
    optimizer.solver_info     = nothing
    optimizer.solver_solution = nothing
    optimizer.solver_nvars    = nothing
    optimizer.sense = MOI.MIN_SENSE # model parameter, so needs to be reset
    optimizer.objconstant = zero(T)
    optimizer.rowranges = UnitRange{DefaultInt}[]
end

MOI.is_empty(optimizer::Optimizer{T}) where {T} = isnothing(optimizer.solver)

function MOI.optimize!(optimizer::Optimizer{T}) where {T}
    if(optimizer.solver_module === Clarabel)
        solution = Clarabel.solve!(optimizer.solver)
    else
        solution = optimizer.solver_module.solve!(optimizer.solver)
    end
    optimizer.solver_solution = solution
    optimizer.solver_info     = optimizer.solver_module.get_info(optimizer.solver)
    nothing
end

function Base.show(io::IO, optimizer::Optimizer{T}) where {T}

    myname = MOI.get(optimizer, MOI.SolverName())
    if isnothing(optimizer.solver)
        print(io,"Empty $(myname) - Optimizer")

    else
        println(io, "$(myname) - Optimizer")
        println(io, " : Has results: $(isnothing(optimizer.solver_solution))")
        println(io, " : Objective constant: $(optimizer.objconstant)")
        println(io, " : Sense: $(optimizer.sense)")
        println(io, " : Precision: $T")

        if !isnothing(optimizer.solver_solution)
        println(io, " : Problem status: $(MOI.get(optimizer,MOI.RawStatusString()))")
        value = round(MOI.get(optimizer,MOI.ObjectiveValue()),digits=3)
        println(io, " : Optimal objective: $(value)")
        println(io, " : Iterations: $(MOI.get(optimizer,MOI.BarrierIterations()))")
        setuptime = round.(optimizer.solver_info.setup_phase_time*1000,digits=2)
        solvetime = round.(optimizer.solver_info.solve_phase_time*1000,digits=2)
        totaltime = round.(optimizer.solver_info.solve_time*1000,digits=2)
        println(io, " : Total time: $(totaltime)ms (", "setup time: $(setuptime)ms, ", "solve time: $(solvetime)ms", ")")
        end
    end
end


#-----------------------------
# Solver Attributes, get/set
#-----------------------------

MOI.get(opt::Optimizer, ::MOI.SolverName)        = string(opt.solver_module)
MOI.get(opt::Optimizer, ::MOI.SolverVersion)     = Clarabel.version()
MOI.get(opt::Optimizer, ::MOI.RawSolver)         = opt.solver
MOI.get(opt::Optimizer, ::MOI.ResultCount)       = DefaultInt(!isnothing(opt.solver_solution))
MOI.get(opt::Optimizer, ::MOI.NumberOfVariables) = opt.solver_nvars
MOI.get(opt::Optimizer, ::MOI.SolveTimeSec)      = opt.solver_info.solve_time
MOI.get(opt::Optimizer, ::MOI.RawStatusString)   = string(opt.solver_info.status)
MOI.get(opt::Optimizer, ::MOI.BarrierIterations) = DefaultInt(opt.solver_info.iterations)

function MOI.get(opt::Optimizer, a::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(opt, a)
    rawobj = opt.solver_info.cost_primal + opt.objconstant
    return opt.sense == MOI.MIN_SENSE ? rawobj : -rawobj
end

function MOI.get(opt::Optimizer, a::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(opt, a)
    rawobj = opt.solver_info.cost_dual + opt.objconstant
    return opt.sense == MOI.MIN_SENSE ? rawobj : -rawobj
end

MOI.supports(::Optimizer, ::MOI.TerminationStatus) = true
function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    !isnothing(opt.solver_solution) || return MOI.OPTIMIZE_NOT_CALLED
    return ClarabeltoMOITerminationStatus[opt.solver_info.status]
end

MOI.supports(::Optimizer, ::MOI.PrimalStatus) = true
function MOI.get(opt::Optimizer, attr::MOI.PrimalStatus)
    if isnothing(opt.solver_solution) || attr.result_index != 1
        return MOI.NO_SOLUTION
    else
        return ClarabeltoMOIPrimalStatus[opt.solver_info.status]
    end
end

MOI.supports(::Optimizer, a::MOI.DualStatus) = true
function MOI.get(opt::Optimizer, attr::MOI.DualStatus)
    if isnothing(opt.solver_solution) || attr.result_index != 1
        return MOI.NO_SOLUTION
    else
        return ClarabeltoMOIDualStatus[opt.solver_info.status]
    end
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.get(opt::Optimizer, ::MOI.Silent) = !opt.solver_settings.verbose
MOI.set(opt::Optimizer, ::MOI.Silent, v::Bool) = (opt.solver_settings.verbose = !v)


MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.get(opt::Optimizer, param::MOI.RawOptimizerAttribute)

    #catch wrapper level attributes.  #otherwise pass to solver
    if(param.name ==  "use_quad_obj")
        return opt.use_quad_obj
    else
        return getproperty(opt.solver_settings, Symbol(param.name))
    end
end


function MOI.set(opt::Optimizer, param::MOI.RawOptimizerAttribute, value) 

    #catch wrapper level attributes.  #otherwise pass to solver
    if(param.name ==  "use_quad_obj")
        opt.use_quad_obj = value
    else
        setproperty!(opt.solver_settings, Symbol(param.name), value)
    end
end

MOI.supports(::Optimizer, ::MOI.VariablePrimal) = true
function MOI.get(opt::Optimizer, a::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(opt, a)
    return opt.solver_solution.x[vi.value]
end

MOI.supports(::Optimizer, ::MOI.ConstraintPrimal) = true
function MOI.get(
    opt::Optimizer,
      a::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{F, S}
) where {F, S <: MOI.AbstractSet}

    MOI.check_result_index_bounds(opt, a)
    rows = constraint_rows(opt.rowranges, ci)
    return opt.solver_solution.s[rows]
end

MOI.supports(::Optimizer, ::MOI.ConstraintDual) = true
function MOI.get(
    opt::Optimizer,
      a::MOI.ConstraintDual,
     ci::MOI.ConstraintIndex{F, S}
) where {F, S <: MOI.AbstractSet}

    MOI.check_result_index_bounds(opt, a)
    rows = constraint_rows(opt.rowranges, ci)
    return opt.solver_solution.z[rows]
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
function MOI.get(opt::Optimizer, ::MOI.TimeLimitSec)
    return MOI.get(opt, MOI.RawOptimizerAttribute("time_limit"))
end
function MOI.set(opt::Optimizer{T}, ::MOI.TimeLimitSec, value::Real) where {T}
    MOI.set(opt, MOI.RawOptimizerAttribute("time_limit"), T(value))
end
function MOI.set(opt::Optimizer{T}, attr::MOI.TimeLimitSec, ::Nothing) where {T}
    MOI.set(opt, MOI.RawOptimizerAttribute("time_limit"), T(Inf))
end


MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = true
function MOI.get(model::Optimizer, ::MOI.NumberOfThreads)::Int
    return MOI.get(model, MOI.RawOptimizerAttribute("max_threads"))
end
function MOI.set(model::Optimizer, ::MOI.NumberOfThreads, threads::Int)
    MOI.set(model, MOI.RawOptimizerAttribute("max_threads"), threads)
    return
end


#------------------------------
# supported constraint types
#------------------------------

function MOI.supports_constraint(
    opt::Optimizer{T},
    ::Type{<:MOI.VectorAffineFunction{T}},
    t::Type{<:OptimizerSupportedMOICones{T}}
) where{T}  
    true
end


function MOI.supports_constraint(
    opt::Optimizer{T},
    ::Type{<:MOI.VectorOfVariables},
    t::Type{<:OptimizerSupportedMOICones{T}}
) where {T}     
    return true
end


#------------------------------
# supported objective functions
#------------------------------

function MOI.supports(
    opt::Optimizer{T},
    ::MOI.ObjectiveFunction{<:Union{
       MOI.ScalarAffineFunction{T},
    }}
) where {T} 
    true
end

function MOI.supports(
    opt::Optimizer{T},
    ::MOI.ObjectiveFunction{<:Union{
       MOI.ScalarQuadraticFunction{T},
    }}
) where {T}
    opt.use_quad_obj
end


#------------------------------
# copy_to interface
#------------------------------

#NB: this solver does *not* support MOI incremental interface

function MOI.copy_to(dest::Optimizer{T}, src::MOI.ModelLike) where {T}

    # generate buckets first (cone grouping order)
    buckets = bucket_constraints_by_cone(src)

    # build idxmap with constraint ids in cone grouping order
    idxmap = MOIU.IndexMap(dest, src, buckets)

    #check all model/variable/constraint attributes to
    #ensure that everything passed is handled by the solver
    copy_to_check_attributes(dest,src)

    # rowranges length = number of constraints in dest indexing
    ncon = 0
    for (_F, _S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        ncon += length(MOI.get(src, MOI.ListOfConstraintIndices{_F, _S}()))
    end
    dest.rowranges = Vector{UnitRange{DefaultInt}}(undef, ncon)

    #assemble the constraints data
    assign_constraint_row_ranges!(dest.rowranges, idxmap, src, buckets)
    A, b, cone_spec, = process_constraints(dest, src, idxmap, buckets)

    #assemble the objective data
    dest.sense = MOI.get(src, MOI.ObjectiveSense())
    P, q, dest.objconstant = process_objective(dest, src, idxmap)

    #Just make a fresh solver with this data, using whatever
    #solver module is configured.   The module will be either
    #Clarabel or ClarabelRs
    dest.solver_nvars = length(q)
    dest.solver = dest.solver_module.Solver(P,q,A,b,cone_spec,dest.solver_settings)

    return idxmap
end

function copy_to_check_attributes(dest, src)

    #allowable model attributes
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        if attr == MOI.Name()           ||
           attr == MOI.ObjectiveSense() ||
           attr isa MOI.ObjectiveFunction
            continue
        end
        throw(MOI.UnsupportedAttribute(attr))
    end

    #allowable variable attributes
    for attr in MOI.get(src, MOI.ListOfVariableAttributesSet())
        if attr == MOI.VariableName()
            continue
        end
        throw(MOI.UnsupportedAttribute(attr))
    end

    #allowable constraint types and attributes
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if !MOI.supports_constraint(dest, F, S)
            throw(MOI.UnsupportedConstraint{F, S}())
        end
        for attr in MOI.get(src, MOI.ListOfConstraintAttributesSet{F, S}())
            if attr == MOI.ConstraintName()
                continue
            end
            throw(MOI.UnsupportedAttribute(attr))
        end
    end

    return nothing
end



#Set up index map from `src` variables and constraints to `dest` variables and constraints.

function MOIU.IndexMap(dest::Optimizer, src::MOI.ModelLike, buckets)
    idxmap = MOIU.IndexMap()

    vis_src = MOI.get(src, MOI.ListOfVariableIndices())
    for i in eachindex(vis_src)
        idxmap[vis_src[i]] = MOI.VariableIndex(i)
    end

    i = 0
    function assign_bucket!(bucket)
        for (F,S,cis) in bucket
            for ci in cis
                i += 1
                idxmap[ci] = MOI.ConstraintIndex{F,S}(i)
            end
        end
    end
    
    #Ensure the correct ordering among different types of cones
    assign_bucket!(buckets.zeros)
    assign_bucket!(buckets.nonneg)
    assign_bucket!(buckets.soc)
    assign_bucket!(buckets.exp)
    assign_bucket!(buckets.power)
    assign_bucket!(buckets.psd)
    assign_bucket!(buckets.genpow)

    return idxmap
end

# ------------------------------
# NEW: bucket constraints by cone type (preserves grouping order)
# ------------------------------
const BucketTriple = Tuple{Type, Type, Vector{MOI.ConstraintIndex}}

function bucket_constraints_by_cone(src::MOI.ModelLike)
    buckets = (
        zeros   = BucketTriple[],
        nonneg  = BucketTriple[],
        soc     = BucketTriple[],
        exp     = BucketTriple[],
        power   = BucketTriple[],
        psd     = BucketTriple[],
        genpow  = BucketTriple[],
    )

    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        cis = MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
        isempty(cis) && continue

        if S <: MOI.Zeros
            push!(buckets.zeros, (F, S, cis))
        elseif S <: MOI.Nonnegatives
            push!(buckets.nonneg, (F, S, cis))
        elseif S <: MOI.SecondOrderCone
            push!(buckets.soc, (F, S, cis))
        elseif S <: MOI.ExponentialCone
            push!(buckets.exp, (F, S, cis))
        elseif S <: MOI.PowerCone
            push!(buckets.power, (F, S, cis))
        elseif S <: MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle}
            push!(buckets.psd, (F, S, cis))
        elseif S <: Clarabel.MOI.GenPowerCone
            push!(buckets.genpow, (F, S, cis))
        else
            throw(MOI.UnsupportedConstraint{F, S}())
        end
    end

    return buckets
end

# ------------------------------
# Row ranges (bucketed, preserves order)
# ------------------------------
function assign_constraint_row_ranges!(
    rowranges::Vector{UnitRange{DefaultInt}},
    idxmap::MOIU.IndexMap,
    src::MOI.ModelLike,
    buckets
)
    startrow = DefaultInt(1)

    function process_bucket!(bucket)
        for (_F, _S, cis) in bucket
            for ci_src in cis
                set = MOI.get(src, MOI.ConstraintSet(), ci_src)
                ci_dest = idxmap[ci_src]
                dim = DefaultInt(MOI.dimension(set))
                endrow = startrow + dim - 1
                rowranges[ci_dest.value] = startrow:endrow
                startrow = endrow + 1
            end
        end
        return nothing
    end

    process_bucket!(buckets.zeros)
    process_bucket!(buckets.nonneg)
    process_bucket!(buckets.soc)
    process_bucket!(buckets.exp)
    process_bucket!(buckets.power)
    process_bucket!(buckets.psd)
    process_bucket!(buckets.genpow)

    return nothing
end

# ------------------------------
# Row range helpers
# ------------------------------
function constraint_rows(
    rowranges::Vector{UnitRange{DefaultInt}},
    ci::MOI.ConstraintIndex{<:Any, <:MOI.AbstractScalarSet},
)
    rowrange = rowranges[ci.value]
    length(rowrange) == 1 || error("Scalar set had dimension != 1")
    return first(rowrange)
end

constraint_rows(
    rowranges::Vector{UnitRange{DefaultInt}},
    ci::MOI.ConstraintIndex{<:Any, <:MOI.AbstractVectorSet},
) = rowranges[ci.value]

constraint_rows(optimizer::Optimizer, ci::MOI.ConstraintIndex) = constraint_rows(optimizer.rowranges, ci)

# -------------------
# Constraint assembly
# -------------------
function process_constraints(
    dest::Optimizer{T}, 
    src::MOI.ModelLike, 
    idxmap,
    buckets
) where {T}

    rowranges = dest.rowranges
    m = isempty(rowranges) ? 0 : last(rowranges[end])
    b = zeros(T, m)

    #these will be used for a triplet representation of A
    nnz = calculate_nnz(src, buckets)
    I = sizehint!(DefaultInt[], nnz)
    J = sizehint!(DefaultInt[], nnz)
    V = sizehint!(T[], nnz)

    #these will be used for the Clarabel API cone types
    cone_spec = sizehint!(Clarabel.SupportedCone[],length(rowranges))

    push_constraint!(
        (I, J, V), b, cone_spec,
        src, idxmap, rowranges, buckets)

    #we have built Ax + b \in Cone, but we actually
    #want to pose the problem as Ax + s = b, s\ in Cone
    V .= -V  #changes sign of A

    n = MOI.get(src, MOI.NumberOfVariables())
    A = sparse(I, J, V, m, n)

    return (A, b, cone_spec)

end

function calculate_nnz(src::MOI.ModelLike, buckets)
    nnz = 0

    function scan_bucket!(bucket)
        for (_F, _S, cis) in bucket
            for ci in cis
                f = MOI.get(src, MOI.ConstraintFunction(), ci)
                nnz += calculate_nnz_single(f)
            end
        end
    end

    scan_bucket!(buckets.zeros)
    scan_bucket!(buckets.nonneg)
    scan_bucket!(buckets.soc)
    scan_bucket!(buckets.exp)
    scan_bucket!(buckets.power)
    scan_bucket!(buckets.psd)
    scan_bucket!(buckets.genpow)

    return nnz
end

calculate_nnz_single(f::MOI.VectorAffineFunction{T}) where {T} = length(f.terms)
calculate_nnz_single(f::MOI.VectorOfVariables) = length(f.variables)

# bucketed push_constraint! (preserves grouping order)
function push_constraint!(
    triplet::SparseTriplet,
    b::Vector{T},
    cone_spec::Vector{Clarabel.SupportedCone},
    src::MOI.ModelLike,
    idxmap,
    rowranges::Vector{UnitRange{DefaultInt}},
    buckets
) where {T}

    function process_bucket!(bucket)
        for (_F, _S, cis) in bucket
            # println("cis is ", cis)
            for ci in cis
                s = MOI.get(src, MOI.ConstraintSet(), ci)
                f = MOI.get(src, MOI.ConstraintFunction(), ci)
                rows = constraint_rows(rowranges, idxmap[ci])

                # println("b len is ", length(b))
                # println("row is ", rows)
                push_constraint_constant!(b, rows, f, s)
                push_constraint_linear!(triplet, f, rows, idxmap, s)
                push_constraint_set!(cone_spec, rows, s)
            end
        end
        return nothing
    end

    process_bucket!(buckets.zeros)
    process_bucket!(buckets.nonneg)
    process_bucket!(buckets.soc)
    process_bucket!(buckets.exp)
    process_bucket!(buckets.power)
    process_bucket!(buckets.psd)
    process_bucket!(buckets.genpow)

    return nothing
end

function push_constraint_constant!(
    b::AbstractVector{T},
    rows::UnitRange{DefaultInt},
    f::MOI.VectorAffineFunction{T},
    ::OptimizerSupportedMOICones{T},
) where {T}

    b[rows] .= f.constants
    return nothing
end

function push_constraint_constant!(
    b::AbstractVector{T},
    rows::UnitRange{DefaultInt},
    f::MOI.VectorOfVariables,
    s::OptimizerSupportedMOICones{T},
) where {T}
    b[rows] .= zero(T)
    return nothing
end

function push_constraint_linear!(
    triplet::SparseTriplet,
    f::MOI.VectorAffineFunction{T},
    rows::UnitRange{DefaultInt},
    idxmap,
    s::OptimizerSupportedMOICones{T},
) where {T}
    (I, J, V) = triplet
    for term in f.terms
        row = rows[term.output_index]
        var = term.scalar_term.variable
        coeff = term.scalar_term.coefficient
        col = idxmap[var].value
        push!(I, row)
        push!(J, col)
        push!(V, coeff)
    end
    return nothing
end

function push_constraint_linear!(
    triplet::SparseTriplet{T},
    f::MOI.VectorOfVariables,
    rows::UnitRange{DefaultInt},
    idxmap,
    s::OptimizerSupportedMOICones{T},
) where {T}
    (I, J, V) = triplet
    @inbounds for k in eachindex(f.variables)
        push!(I, rows[k])
        push!(J, idxmap[f.variables[k]].value)
        push!(V, one(T))
    end
    return nothing
end


function push_constraint_set!(
    cone_spec::Vector{Clarabel.SupportedCone},
    rows::Union{DefaultInt,UnitRange{DefaultInt}},
    s::OptimizerSupportedMOICones{T},
) where {T}

    #we need to handle PowerCones differently here because
    # 1) they have a power and not a a dimension (always 3),
    # 2) we can't use [typeof(s)] as a key into MOItoClarabelCones
    # because typeof(s) = MOI.PowerCone{T} and the dictionary
    # has keys with the *unparametrized* types
    if isa(s,MOI.PowerCone)
        pow_cone_type = MOItoClarabelCones[MOI.PowerCone]
        push!(cone_spec, pow_cone_type(s.exponent))
        return nothing
    end

    # handle ExponentialCone differently because it
    # doesn't take dimension as a parameter (always 3)
    if isa(s,MOI.ExponentialCone)
        exp_cone_type = MOItoClarabelCones[MOI.ExponentialCone]
        push!(cone_spec, exp_cone_type())
        return nothing
    end

    # handle GenPowerCone (takes two parameters)
    if isa(s,Clarabel.MOI.GenPowerCone)
        genpow_cone_type = MOItoClarabelCones[Clarabel.MOI.GenPowerCone]
        push!(cone_spec, genpow_cone_type(s.α,s.dim2))
        return nothing
    end

    next_type = MOItoClarabelCones[typeof(s)]
    next_dim  = _to_optimizer_conedim(s)

    # merge cones together where :
    # 1) cones of the same type appear consecutively and
    # 2) those cones are 1-D.
    # This is just the zero and nonnegative cones

    if isempty(cone_spec) || next_type ∉ OptimizerMergeableTypes || next_type != typeof(cone_spec[end])
        push!(cone_spec, next_type(next_dim))
    else
        #overwrite with a a cone of enlarged dimension
        cone_spec[end] = next_type(next_dim + cone_spec[end].dim)
    end

    return nothing
end

# converts number of elements to optimizer's internal dimension parameter.
# For matrices, this is just the matrix side dimension.  Conversion differs
# for square vs triangular form
_to_optimizer_conedim(set::MOI.AbstractVectorSet) = MOI.dimension(set)
_to_optimizer_conedim(set::MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle}) = MOI.side_dimension(set)

function push_constraint_set!(
    cone_spec::Vector{Clarabel.SupportedCone},
    rows::Union{DefaultInt,UnitRange{DefaultInt}},
    s::MathOptInterface.AbstractSet
)
    #landing here means that s ∉ OptimizerSupportedMOICones.
    throw(MOI.UnsupportedConstraint(s))
end


# objective assembly
# -------------------

# Construct cost function data so that minimize `1/2 x' P x + q' x + c`,
# being careful of objective sense

function process_objective(
    dest::Optimizer{T},
    src::MOI.ModelLike,
    idxmap
) where {T}

    sense = dest.sense
    n = MOI.get(src, MOI.NumberOfVariables())

    if sense == MOI.FEASIBILITY_SENSE
        P = spzeros(T, n, n)
        q = zeros(T,n)
        c = T(0.)

    else
        function_type = MOI.get(src, MOI.ObjectiveFunctionType())
        q = zeros(T,n)

        if function_type == MOI.ScalarAffineFunction{T}
            faffine = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}())
            P = spzeros(T, n, n)
            process_objective_linearterm!(q, faffine.terms, idxmap)
            c = faffine.constant

        elseif function_type == MOI.ScalarQuadraticFunction{T}
            fquadratic = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}())
            I = [DefaultInt(idxmap[term.variable_1].value) for term in fquadratic.quadratic_terms]
            J = [DefaultInt(idxmap[term.variable_2].value) for term in fquadratic.quadratic_terms]
            V = [term.coefficient for term in fquadratic.quadratic_terms]
            upper_triangularize!((I, J, V))
            P = sparse(I, J, V, n, n)
            process_objective_linearterm!(q, fquadratic.affine_terms, idxmap)
            c = fquadratic.constant

        else
            throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunction{function_type}()))
        end

        if sense == MOI.MAX_SENSE
            P.nzval .= -P.nzval
            q       .= -q
            c        = -c
        end

    end
    return (P, q, c)
end


function process_objective_linearterm!(
    q::AbstractVector{T},
    terms::Vector{<:MOI.ScalarAffineTerm},
    idxmapfun::Function = identity
) where {T}

    q .= 0
    for term in terms
        var = term.variable
        coeff = term.coefficient
        q[idxmapfun(var).value] += coeff
    end
    return nothing
end

function process_objective_linearterm!(
    q::AbstractVector{T},
    terms::Vector{<:MOI.ScalarAffineTerm},
    idxmap::MOIU.IndexMap
) where {T}
    process_objective_linearterm!(q, terms, var -> idxmap[var])
end

function upper_triangularize!(triplet::SparseTriplet{T}) where {T}

    (I, J, V) = triplet
    n = length(V)
    (length(I) == length(J) == n) || error()
    for i = eachindex(I)
        if I[i] > J[i]
            I[i], J[i] = J[i], I[i]
        end
    end
end
