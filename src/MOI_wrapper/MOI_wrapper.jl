# ======================================================================
#  Clarabel MOI wrapper (bucketed, preserves cone grouping order)
#  Grouping order: Zeros -> Nonnegatives -> SOC -> Exp -> Power -> PSD -> GenPower
# ======================================================================
using MathOptInterface, SparseArrays
using ..Clarabel
export Optimizer, direct_optimizer
using CUDA

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

# =============================
# Cache backend (Zaphod+SCS-style)
# =============================

# Define _SetConstants addressing power cones
struct _SetConstants{T}
    b::Vector{T}
    power_exp::Dict{Int,T}                 # key = first row index
    genpow_params::Dict{Int,Tuple{Vector{T},Int}}  # key = first row index
    _SetConstants{T}() where {T} = new{T}(T[], Dict{Int,T}(), Dict{Int,Tuple{Vector{T},Int}}())
end

function Base.empty!(x::_SetConstants)
    empty!(x.b)
    empty!(x.power_exp)
    empty!(x.genpow_params)
    return x
end

Base.resize!(x::_SetConstants, n) = resize!(x.b, n)

# Important: the order here is the order we will iterate to build cone_spec / rowranges.
MOI.Utilities.@product_of_sets(
    Cones,
    MOI.Zeros,
    MOI.Nonnegatives,
    MOI.SecondOrderCone,
    MOI.ExponentialCone,
    MOI.PowerCone{T},
    MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle},
    Clarabel.MOI.GenPowerCone{T},
)

const OptimizerCache{T} = MOI.Utilities.GenericModel{
    T,
    MOI.Utilities.ObjectiveContainer{T},
    MOI.Utilities.VariablesContainer{T},
    MOI.Utilities.MatrixOfConstraints{
        T,
        MOI.Utilities.MutableSparseMatrixCSC{
            T,
            Int,
            MOI.Utilities.OneBasedIndexing,
        },
        _SetConstants{T},
        Cones{T},
    },
}

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

function MOI.default_cache(::Optimizer{T}, ::Type{T}) where {T}
    return MOI.Utilities.UniversalFallback(OptimizerCache{T}())
end

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
    CUDA.@allowscalar return opt.solver_solution.x[vi.value]
end

MOI.supports(::Optimizer, ::MOI.ConstraintPrimal) = true
function MOI.get(
    opt::Optimizer,
      a::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{F, S}
) where {F, S <: MOI.AbstractSet}

    MOI.check_result_index_bounds(opt, a)
    rows = constraint_rows(opt.rowranges, ci)
    CUDA.@allowscalar return opt.solver_solution.s[rows]
end

MOI.supports(::Optimizer, ::MOI.ConstraintDual) = true
function MOI.get(
    opt::Optimizer,
      a::MOI.ConstraintDual,
     ci::MOI.ConstraintIndex{F, S}
) where {F, S <: MOI.AbstractSet}

    MOI.check_result_index_bounds(opt, a)
    rows = constraint_rows(opt.rowranges, ci)
    CUDA.@allowscalar return opt.solver_solution.z[rows]
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


# Default: load affine function constants into b
function MOI.Utilities.load_constants(x::_SetConstants, offset, f)
    MOI.Utilities.load_constants(x.b, offset, f)
    return
end

# PowerCone stores exponent (one scalar) at the first row of this constraint
function MOI.Utilities.load_constants(
    x::_SetConstants{T},
    offset::Int,
    set::MOI.PowerCone{T},
) where {T}
    x.power_exp[offset + 1] = set.exponent
    return
end

# GenPowerCone: store its parameters at first row
function MOI.Utilities.load_constants(
    x::_SetConstants{T},
    offset::Int,
    set::Clarabel.MOI.GenPowerCone{T},
) where {T}
    x.genpow_params[offset + 1] = (copy(set.α), set.dim2)
    return
end

function MOI.Utilities.set_from_constants(x::_SetConstants, S, rows)
    return MOI.Utilities.set_from_constants(x.b, S, rows)
end

function MOI.Utilities.set_from_constants(
    x::_SetConstants{T},
    ::Type{MOI.PowerCone{T}},
    rows,
) where {T}
    @assert length(rows) == 3
    return MOI.PowerCone{T}(x.power_exp[first(rows)])
end

function MOI.Utilities.set_from_constants(
    x::_SetConstants{T},
    ::Type{Clarabel.MOI.GenPowerCone{T}},
    rows,
) where {T}
    α, dim2 = x.genpow_params[first(rows)]
    return Clarabel.MOI.GenPowerCone(α, dim2)
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

# ---- NEW: block metadata for row reordering for SOCs----
struct _Block{T}
    ci::MOI.ConstraintIndex
    set::Any
    r1::Int
    len::Int
end

# collect blocks for a given set type S (VectorAffineFunction{T}-in-S only)
function _collect_blocks(::Type{S}, src::OptimizerCache{T}) where {S,T}
    blocks = _Block{T}[]
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{T}, S}())
        rows = MOI.Utilities.rows(src.constraints.sets, ci)
        set  = MOI.get(src, MOI.ConstraintSet(), ci)
        push!(blocks, _Block{T}(ci, set, first(rows), length(rows)))
    end
    return blocks
end

# build desired block order:
# Zeros -> Nonneg -> (SOC large then SOC small) -> Exp -> Power -> PSD -> GenPower
function _ordered_blocks_with_soc_split(src::OptimizerCache{T}, n_threshold::Int) where {T}
    blocks = _Block{T}[]

    append!(blocks, _collect_blocks(MOI.Zeros, src))
    append!(blocks, _collect_blocks(MOI.Nonnegatives, src))

    soc_blocks = _collect_blocks(MOI.SecondOrderCone, src)
    small_soc = _Block{T}[]
    large_soc = _Block{T}[]
    for bl in soc_blocks
        if bl.len > n_threshold
            push!(large_soc, bl)
        else
            push!(small_soc, bl)
        end
    end
    append!(blocks, large_soc)
    append!(blocks, small_soc)

    append!(blocks, _collect_blocks(MOI.ExponentialCone, src))

    # NOTE: use unparameterized MOI.PowerCone here; it still matches indices
    append!(blocks, _collect_blocks(MOI.PowerCone{T}, src))

    append!(blocks, _collect_blocks(MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle}, src))

    append!(blocks, _collect_blocks(Clarabel.MOI.GenPowerCone{T}, src))

    return blocks
end

# permute rows of A,b by block order, and rebuild rowranges + cone_spec
function _apply_block_order!(
    dest::Optimizer{T},
    src::OptimizerCache{T},
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    blocks::Vector{_Block{T}},
) where {T}
    m = length(b)

    # build row permutation p: new_row -> old_row
    p = Vector{Int}(undef, m)
    new_r = 0

    for bl in blocks
        for k in 1:bl.len
            p[new_r + k] = bl.r1 + k - 1
        end
        new_r += bl.len
    end
    @assert new_r == m

    # apply permutation
    A2 = A[p, :]
    b2 = b[p]

    # rebuild rowranges indexed by ci.value (sparse index space)
    max_ci = length(dest.rowranges)  # already allocated by caller
    fill!(dest.rowranges, 0: -1)     # mark unfilled (optional)

    cone_spec = Clarabel.SupportedCone[]
    new_r = 1
    for bl in blocks
        r2 = new_r + bl.len - 1
        dest.rowranges[bl.ci.value] = UnitRange{DefaultInt}(new_r, r2)
        _push_cone_spec_merged!(cone_spec, bl.set)
        new_r = r2 + 1
    end

    return A2, b2, cone_spec
end

#------------------------------
# copy_to interface
#------------------------------

#NB: this solver does *not* support MOI incremental interface

function MOI.copy_to(dest::Optimizer{T}, src::MOI.ModelLike) where {T}
    cache = OptimizerCache{T}()
    index_map = MOI.copy_to(cache, src)        # MOI builds A/b efficiently into MutableSparseMatrixCSC
    MOI.copy_to(dest, cache)                   # we extract into Clarabel solver data
    return index_map
end

function MOI.copy_to(
    dest::Optimizer{T},
    src::MOI.Utilities.UniversalFallback{OptimizerCache{T}},
) where {T}
    MOI.Utilities.throw_unsupported(src)
    return MOI.copy_to(dest, src.model)
end

function MOI.copy_to(dest::Optimizer{T}, src::OptimizerCache{T}) where {T}
    MOI.empty!(dest)

    # ---------------------------
    # Extract constraint matrix
    # ---------------------------
    Ab = src.constraints

    # Clarabel uses Ax + s = b with s in Cone.
    # The cache stores "A_cache * x + const in set". Your original wrapper flips sign of A.
    # Keep the same convention:
    A = -convert(SparseMatrixCSC{T,Int}, Ab.coefficients)
    b = copy(Ab.constants.b)

    n = size(A, 2)
    dest.solver_nvars = n

    # ---------------------------
    # Objective: build P, q, c
    # ---------------------------
    dest.sense = MOI.get(src, MOI.ObjectiveSense())
    P, q, c = _process_objective_from_cache(dest, src, n)

    # ---------------------------
    # Build rowranges + cone_spec in your preferred order
    # ---------------------------
    # We also need dest.rowranges indexed by dest ConstraintIndex value.
    # We'll iterate constraint types in the same order as Cones declaration.
    # Allocate rowranges so it is safe to index by ci.value (not dense in GenericModel)
    max_ci = 0
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
            max_ci = max(max_ci, ci.value)
        end
    end
    dest.rowranges = Vector{UnitRange{DefaultInt}}(undef, max_ci)

    blocks = _ordered_blocks_with_soc_split(src, Clarabel.SOC_NO_EXPANSION_MAX_SIZE)

    A, b, cone_spec = _apply_block_order!(dest, src, A, b, blocks)

    # ---------------------------
    # Create solver
    # ---------------------------
    dest.objconstant = c
    dest.solver = dest.solver_module.Solver(P, q, A, b, cone_spec, dest.solver_settings)

    return MOI.Utilities.identity_index_map(src)
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

# converts number of elements to optimizer's internal dimension parameter.
# For matrices, this is just the matrix side dimension.  Conversion differs
# for square vs triangular form
_to_optimizer_conedim(set::MOI.AbstractVectorSet) = MOI.dimension(set)
_to_optimizer_conedim(set::MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle}) = MOI.side_dimension(set)


# objective assembly
# -------------------

# Construct cost function data so that minimize `1/2 x' P x + q' x + c`,
# being careful of objective sense


function _process_objective_from_cache(
    dest::Optimizer{T},
    src::OptimizerCache{T},
    n::Int,
) where {T}
    sense = dest.sense

    if sense == MOI.FEASIBILITY_SENSE
        return (spzeros(T, n, n), zeros(T, n), zero(T))
    end

    ftype = MOI.get(src, MOI.ObjectiveFunctionType())

    if ftype == MOI.ScalarAffineFunction{T}
        f = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}())
        P = spzeros(T, n, n)
        q = zeros(T, n)
        for term in f.terms
            q[term.variable.value] += term.coefficient
        end
        c = f.constant
        if sense == MOI.MAX_SENSE
            q .= -q
            c = -c
        end
        return (P, q, c)

    elseif ftype == MOI.ScalarQuadraticFunction{T}
        f = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}())
        q = zeros(T, n)
        for term in f.affine_terms
            q[term.variable.value] += term.coefficient
        end
        I = Int[]
        J = Int[]
        V = T[]
        sizehint!(I, length(f.quadratic_terms))
        sizehint!(J, length(f.quadratic_terms))
        sizehint!(V, length(f.quadratic_terms))
        for qt in f.quadratic_terms
            i = qt.variable_1.value
            j = qt.variable_2.value
            if i > j
                i, j = j, i
            end
            push!(I, i); push!(J, j); push!(V, qt.coefficient)
        end
        P = sparse(I, J, V, n, n)
        c = f.constant
        if sense == MOI.MAX_SENSE
            P.nzval .= -P.nzval
            q .= -q
            c = -c
        end
        return (P, q, c)
    else
        throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunction{ftype}()))
    end
end

# function _append_cones_and_rowranges!(
#     dest::Optimizer{T},
#     src::OptimizerCache{T},
#     cone_spec::Vector{Clarabel.SupportedCone},
# ) where {T}

#     function process_one_settype!(S)
#         for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{T}, S}())
#             rows = MOI.Utilities.rows(src.constraints.sets, ci)
#             dest.rowranges[ci.value] = UnitRange{DefaultInt}(first(rows), last(rows))
#             set = MOI.get(src, MOI.ConstraintSet(), ci)
#             _push_cone_spec_merged!(cone_spec, set)
#         end
#         return nothing
#     end

#     process_one_settype!(MOI.Zeros)
#     process_one_settype!(MOI.Nonnegatives)
#     process_one_settype!(MOI.SecondOrderCone)
#     process_one_settype!(MOI.ExponentialCone)
#     process_one_settype!(MOI.PowerCone{T})
#     process_one_settype!(MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle})
#     process_one_settype!(Clarabel.MOI.GenPowerCone{T})

#     return nothing
# end

function _push_cone_spec_merged!(cone_spec::Vector{Clarabel.SupportedCone}, s)
    # Power cone special casing
    if isa(s, MOI.PowerCone)
        pow_cone_type = MOItoClarabelCones[MOI.PowerCone]
        push!(cone_spec, pow_cone_type(s.exponent))
        return nothing
    end

    # Exponential cone special casing
    if isa(s, MOI.ExponentialCone)
        exp_cone_type = MOItoClarabelCones[MOI.ExponentialCone]
        push!(cone_spec, exp_cone_type())
        return nothing
    end

    # GenPower special casing
    if isa(s, Clarabel.MOI.GenPowerCone)
        genpow_cone_type = MOItoClarabelCones[Clarabel.MOI.GenPowerCone]
        push!(cone_spec, genpow_cone_type(s.α, s.dim2))
        return nothing
    end

    next_type = MOItoClarabelCones[typeof(s)]
    next_dim  = _to_optimizer_conedim(s)

    # Merge only Zero/Nonneg consecutive blocks 
    if isempty(cone_spec) || next_type ∉ OptimizerMergeableTypes || next_type != typeof(cone_spec[end])
        push!(cone_spec, next_type(next_dim))
    else
        cone_spec[end] = next_type(next_dim + cone_spec[end].dim)
    end
    return nothing
end