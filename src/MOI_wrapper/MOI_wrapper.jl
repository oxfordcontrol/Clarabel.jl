using MathOptInterface
export Optimizer

#-----------------------------
# Const definitions
#-----------------------------

const MOI = MathOptInterface
const MOIU = MOI.Utilities
const SparseTriplet{T} = Tuple{Vector{<:Integer}, Vector{<:Integer}, Vector{T}}

#These enumerate the cones supported by the solver and the mapping
#from MOI Cones to IPSolver internal cone types
#NB: When a parametric type like PowerCone{T} is eventually added
#then this should become OptimizerSupportedMOICones{T <: Real}
#and made parametric in supporting functions as well.  It does not
#seem possible to make it this way now since none of the elements
#in the Union are parametric
const OptimizerSupportedMOICones = Union{
    MOI.Zeros,
    MOI.Nonnegatives,
    MOI.SecondOrderCone,
    # MOI.PowerCone{T},
    # MOI.DualPowerCone{T},
    # MOI.PositiveSemidefiniteConeTriangle,
    # MOI.ExponentialCone,
    }

#Optimizer will consolidate cones of these types
#when it is possible to do so
const OptimizerMergeableTypes = [IPSolver.ZeroConeT, IPSolver.NonnegativeConeT]

#mappings between MOI and internal definitions

const MOItoIPSolverCones = Dict([
    MOI.Zeros           => IPSolver.ZeroConeT,
    MOI.Nonnegatives    => IPSolver.NonnegativeConeT,
    MOI.SecondOrderCone => IPSolver.SecondOrderConeT,
])

const IPSolvertoMOITerminationStatus = Dict([
    IPSolver.SOLVED             =>  MOI.OPTIMAL,
    IPSolver.MAX_ITERATIONS     =>  MOI.ITERATION_LIMIT,
    IPSolver.MAX_TIME           =>  MOI.TIME_LIMIT,
    IPSolver.PRIMAL_INFEASIBLE  =>  MOI.INFEASIBLE,
    IPSolver.DUAL_INFEASIBLE    =>  MOI.DUAL_INFEASIBLE
])

const IPSolvertoMOIPrimalStatus = Dict([
    IPSolver.SOLVED             =>  MOI.FEASIBLE_POINT,
    IPSolver.MAX_ITERATIONS     =>  MOI.NEARLY_FEASIBLE_POINT,
    IPSolver.MAX_TIME           =>  MOI.NEARLY_FEASIBLE_POINT,
    IPSolver.PRIMAL_INFEASIBLE  =>  MOI.INFEASIBLE_POINT,
    IPSolver.DUAL_INFEASIBLE    =>  MOI.INFEASIBILITY_CERTIFICATE
])

const IPSolvertoMOIDualStatus = Dict([
    IPSolver.SOLVED             =>  MOI.FEASIBLE_POINT,
    IPSolver.MAX_ITERATIONS     =>  MOI.NEARLY_FEASIBLE_POINT,
    IPSolver.MAX_TIME           =>  MOI.NEARLY_FEASIBLE_POINT,
    IPSolver.PRIMAL_INFEASIBLE  =>  MOI.INFEASIBILITY_CERTIFICATE,
    IPSolver.DUAL_INFEASIBLE    =>  MOI.INFEASIBLE_POINT
])



#-----------------------------
# Main interface struct
#-----------------------------

mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    inner::IPSolver.Solver{T}
    has_results::Bool
    is_empty::Bool
    sense::MOI.OptimizationSense
    objconstant::T
    rowranges::Dict{Int, UnitRange{Int}}

    function Optimizer{T}(; user_settings...) where {T <: AbstractFloat}
        inner = IPSolver.Solver{T}()
        has_results = false
        is_empty = true
        sense = MOI.MIN_SENSE
        objconstant = zero(T)
        rowranges = Dict{Int, UnitRange{Int}}()

        for (key, value) in user_settings
            MOI.set(optimizer, MOI.RawOptimizerAttribute(key), value)
        end
        new(inner,has_results,is_empty,sense,objconstant,rowranges)
    end
end

Optimizer(args...; kwargs...) = Optimizer{DefaultFloat}(args...; kwargs...)


#-----------------------------
# Required basic methods
#-----------------------------

# reset the optimizer
function MOI.empty!(optimizer::Optimizer{T}) where {T <: AbstractFloat}
    #just make a new solveropt, keeping current settings
    optimizer.inner = IPSolver.Solver(optimizer.inner.settings)
    optimizer.has_results = false
    optimizer.is_empty = true
    optimizer.sense = MOI.MIN_SENSE # model parameter, so needs to be reset
    optimizer.objconstant = zero(T)
    optimizer.rowranges = Dict{Int, UnitRange{Int}}()
end

MOI.is_empty(optimizer::Optimizer) = optimizer.is_empty

function MOI.optimize!(optimizer::Optimizer)
    IPSolver.solve!(optimizer.inner)
    optimizer.has_results = true
    nothing
end

function Base.show(io::IO, optimizer::Optimizer{T}) where {T}

    myname = MOI.get(optimizer, MOI.SolverName())
    if optimizer.is_empty
        print(io,"Empty $(myname) - Optimizer")

    else
        println(io, "$(myname) - Optimizer")
        println(io, " : Has results: $(optimizer.has_results)")
        println(io, " : Objective constant: $(optimizer.objconstant)")
        println(io, " : Sense: $(optimizer.sense)")
        println(io, " : Precision: $T")

        if optimizer.has_results
        println(io, " : Problem status: $(MOI.get(optimizer,MOI.RawStatusString()))")
        value = round(MOI.get(optimizer,MOI.ObjectiveValue()),digits=3)
        println(io, " : Optimal objective: $(value)")
        println(io, " : Iterations: $(MOI.get(optimizer,MOI.BarrierIterations()))")
        solvetime = round.(optimizer.inner.info.solve_time*1000,digits=2)
        println(io, " : Solve time: $(solvetime)ms")
        end
    end
end


#-----------------------------
# other methods
#-----------------------------


function set_user_settings!(inner::IPSolver.Solver, user_settings)
    for (k, v) in user_settings
        !in(k, fieldnames(typeof(inner.settings))) && error("The user setting $(k) is not defined.")
        setfield!(inner.settings, k, v)
    end
end


#-----------------------------
# Solver Attributes, get/set
#-----------------------------

MOI.get(opt::Optimizer, ::MOI.SolverName)        = IPSolver.solver_name()
MOI.get(opt::Optimizer, ::MOI.SolverVersion)     = IPSolver.version()
MOI.get(opt::Optimizer, ::MOI.RawSolver)         = opt.inner
MOI.get(opt::Optimizer, ::MOI.ResultCount)       = opt.has_results ? 1 : 0
MOI.get(opt::Optimizer, ::MOI.NumberOfVariables) = opt.inner.data.n
MOI.get(opt::Optimizer, ::MOI.SolveTimeSec)      = opt.inner.info.solve_time
MOI.get(opt::Optimizer, ::MOI.RawStatusString)   = string(opt.inner.info.status)
MOI.get(opt::Optimizer, ::MOI.BarrierIterations) = opt.inner.info.iterations

function MOI.get(opt::Optimizer, a::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(opt, a)
    rawobj = opt.inner.info.cost_primal + opt.objconstant
    return opt.sense == MOI.MIN_SENSE ? rawobj : -rawobj
end

function MOI.get(opt::Optimizer, a::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(opt, a)
    rawobj = opt.inner.info.cost_dual + opt.objconstant
    return opt.sense == MOI.MIN_SENSE ? rawobj : -rawobj
end

MOI.supports(::Optimizer, ::MOI.TerminationStatus) = true
function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    opt.has_results || return MOI.OPTIMIZE_NOT_CALLED
    return IPSolvertoMOITerminationStatus[opt.inner.info.status]
end

MOI.supports(::Optimizer, ::MOI.PrimalStatus) = true
function MOI.get(opt::Optimizer, ::MOI.PrimalStatus)
    opt.has_results || return MOI.NO_SOLUTION
    return IPSolvertoMOIPrimalStatus[opt.inner.info.status]
end

MOI.supports(::Optimizer, ::MOI.DualStatus) = true
function MOI.get(opt::Optimizer, ::MOI.DualStatus)
    opt.has_results || return MOI.NO_SOLUTION
    return IPSolvertoMOIDualStatus[opt.inner.info.status]
end

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.get(opt::Optimizer, ::MOI.Silent) = !opt.inner.settings.verbose
MOI.set(opt::Optimizer, ::MOI.Silent, v::Bool) = (opt.inner.settings.verbose = !v)


MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true
MOI.get(opt::Optimizer, param::MOI.RawOptimizerAttribute) =
    getproperty(opt.inner.settings, Symbol(param.name))
MOI.set(opt::Optimizer, param::MOI.RawOptimizerAttribute, value) =
    setproperty!(opt.inner.settings, Symbol(param.name), value)

MOI.supports(::Optimizer, ::MOI.VariablePrimal) = true
function MOI.get(opt::Optimizer, a::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(opt, a)
    return opt.inner.variables.x[vi.value]
end

MOI.supports(::Optimizer, ::MOI.ConstraintPrimal) = true
function MOI.get(opt::Optimizer, a::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex)
    MOI.check_result_index_bounds(opt, a)
    rows = constraint_rows(opt.rowranges, ci)
    return opt.inner.variables.s[rows]
end

MOI.supports(::Optimizer, ::MOI.ConstraintDual) = true
function MOI.get(opt::Optimizer, a::MOI.ConstraintDual, ci::MOI.ConstraintIndex)
    MOI.check_result_index_bounds(opt, a)
    rows = constraint_rows(opt.rowranges, ci)
    return opt.inner.variables.z[rows]
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
function MOI.get(opt::Optimizer, ::MOI.TimeLimitSec)
    return MOI.get(opt, MOI.RawOptimizerAttribute("time_limit"))
end
function MOI.set(opt::Optimizer{T}, ::MOI.TimeLimitSec, value::Real) where {T}
    MOI.set(opt, MOI.RawOptimizerAttribute("time_limit"), T(value))
end
function MOI.set(opt::Optimizer{T}, attr::MOI.TimeLimitSec, ::Nothing) where {T}
    MOI.set(opt, MOI.RawOptimizerAttribute("time_limit"), zero(T))
end


#attributes not currently supported
MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = false


#------------------------------
# supported constraint types
#------------------------------

MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:MOI.VectorAffineFunction{T}},
    ::Type{<:OptimizerSupportedMOICones}  #will be {T} if parametric cones are added
) where {T} = true

MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:MOI.VectorOfVariables},
    ::Type{<:OptimizerSupportedMOICones}  #will be {T} if parametric cones are added
) where {T} = true


#------------------------------
# supported objective functions
#------------------------------

MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{<:Union{
       MOI.ScalarAffineFunction,
       MOI.ScalarQuadraticFunction,
    }}
) = true


#------------------------------
# copy_to interface
#------------------------------

#NB: this solver does *not* support MOI incremental interface

function MOI.copy_to(dest::Optimizer{T}, src::MOI.ModelLike) where {T}

    idxmap = MOIU.IndexMap(dest, src)

    #check all model/variable/constraint attributes to
    #ensure that everything passed is handled by the solver
    copy_to_check_attributes(dest,src)

    #assemble the constraints data
    assign_constraint_row_ranges!(dest.rowranges, idxmap, src)
    A, b, cone_types, cone_dims = process_constraints(dest, src, idxmap)

    #assemble the objective data
    dest.sense = MOI.get(src, MOI.ObjectiveSense())
    P, q, dest.objconstant = process_objective(dest, src, idxmap)

    #call setup! again on the solver.   This will flush all
    #internal data but will keep settings intact
    IPSolver.setup!(dest.inner,P,q,A,b,cone_types,cone_dims)

    #model is no longer empty
    dest.is_empty = false

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

function MOIU.IndexMap(dest::Optimizer, src::MOI.ModelLike)

    idxmap = MOIU.IndexMap()

    vis_src = MOI.get(src, MOI.ListOfVariableIndices())
    for i in eachindex(vis_src)
        idxmap[vis_src[i]] = MOI.VariableIndex(i)
    end
    i = 0
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        MOI.supports_constraint(dest, F, S) || throw(MOI.UnsupportedConstraint{F, S}())
        cis_src = MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
        for ci in cis_src
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{F, S}(i)
        end
    end

    return idxmap
end


function assign_constraint_row_ranges!(
    rowranges::Dict{Int, UnitRange{Int}},
    idxmap::MOIU.IndexMap,
    src::MOI.ModelLike
)

    startrow = 1
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        cis_src = MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
        for ci_src in cis_src
            set = MOI.get(src, MOI.ConstraintSet(), ci_src)
            ci_dest = idxmap[ci_src]
            endrow = startrow + MOI.dimension(set) - 1
            rowranges[ci_dest.value] = startrow : endrow
            startrow = endrow + 1
        end
    end

    return nothing
end

function constraint_rows(
    rowranges::Dict{Int, UnitRange{Int}},
    ci::MOI.ConstraintIndex{<:Any, <:MOI.AbstractScalarSet}
)
    rowrange = rowranges[ci.value]
    length(rowrange) == 1 || error()
    first(rowrange)
end

constraint_rows(
    rowranges::Dict{Int, UnitRange{Int}},
    ci::MOI.ConstraintIndex{<:Any, <:MOI.AbstractVectorSet}
) = rowranges[ci.value]

constraint_rows(
    optimizer::Optimizer,
    ci::MOI.ConstraintIndex
) = constraint_rows(optimizer.rowranges, ci)


# constraint assembly
# -------------------

#construct constraint data as a single collection of
#constraints Ax + s = b, s \in K, where K is composed
#of the various cones supported by the solver

function process_constraints(
    dest::Optimizer{T},
    src::MOI.ModelLike,
    idxmap
) where {T}

    rowranges = dest.rowranges
    m = mapreduce(length, +, values(rowranges), init=0)
    b = Vector{Float64}(undef, m)

    #these will be used for a triplet representation of A
    I = Int[]
    J = Int[]
    V = T[]

    #these will be used for the IPSolver cone types and dimensions
    cone_types = IPSolver.SupportedCones[]
    cone_dims  = Int[]

    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        push_constraint!(
            (I, J, V), b,
            cone_types, cone_dims,
            src, idxmap, rowranges, F, S)
    end

    #we have built Ax + b \in Cone, but we actually
    #want to pose the problem as Ax + s = b, s\ in Cone
    V .= -V  #changes sign of A

    n = MOI.get(src, MOI.NumberOfVariables())
    A = sparse(I, J, V, m, n)

    return (A, b, cone_types, cone_dims)

end

function push_constraint!(
    triplet::SparseTriplet,
    b::Vector{T},
    cone_types::Vector{IPSolver.SupportedCones},
    cone_dims::Vector{Int},
    src::MOI.ModelLike,
    idxmap,
    rowranges::Dict{Int, UnitRange{Int}},
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet}
) where {T}

    cis_src = MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
    for ci in cis_src
        s = MOI.get(src, MOI.ConstraintSet(), ci)
        f = MOI.get(src, MOI.ConstraintFunction(), ci)
        rows = constraint_rows(rowranges, idxmap[ci])
        push_constraint_constant!(b, rows, f)
        push_constraint_linear!(triplet, f, rows, idxmap)
        push_constraint_set!(cone_types, cone_dims, rows, s)
    end

    return nothing
end


function push_constraint_constant!(
    b::AbstractVector{T},
    rows::UnitRange{Int},
    f::MOI.VectorAffineFunction{T}
) where {T}

    for (i, row) in enumerate(rows)
        b[row] = f.constants[i]
    end
    return nothing
end

function push_constraint_constant!(
    b::AbstractVector{T},
    rows::UnitRange{Int},
    f::MOI.VectorOfVariables
) where {T}
    b[rows] .= zero(T)
    return nothing
end

function push_constraint_linear!(
    triplet::SparseTriplet,
    f::MOI.VectorAffineFunction{T},
    rows::UnitRange{Int},
    idxmap
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
    rows::UnitRange{Int},
    idxmap
) where {T}

    (I, J, V) = triplet
    cols = [idxmap[var].value for var in f.variables]
    append!(I, rows)
    append!(J, cols)
    append!(V, ones(T,length(cols)))

    return nothing
end


function push_constraint_set!(
    cone_types::Vector{IPSolver.SupportedCones},
    cone_dims::Vector{Int},
    rows::Union{Int,UnitRange{Int}},
    s::OptimizerSupportedMOICones #will be {T} if parametric cones are added
) where {T}

    # merge cones together where :
    # 1) cones of the same type appear consecutively and
    # 2) those cones are 1-D.
    # This is just the zero and nonnegative cones

    next_type = MOItoIPSolverCones[typeof(s)]
    next_dim  = length(rows)

    if isempty(cone_types) || next_type ∉ OptimizerMergeableTypes || next_type != cone_types[end]
        push!(cone_types, next_type)
        push!(cone_dims, next_dim)
    else
        cone_dims[end] += next_dim
    end

    return nothing
end

function push_constraint_set!(
    cone_types::Vector{IPSolver.SupportedCones},
    cone_dims::Vector{Int},
    rows::Union{Int,UnitRange{Int}},
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
            P = spzeros(n, n)
            process_objective_linearterm!(q, faffine.terms, idxmap)
            c = faffine.constant

        elseif function_type == MOI.ScalarQuadraticFunction{T}
            fquadratic = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}())
            I = [Int(idxmap[term.variable_1].value) for term in fquadratic.quadratic_terms]
            J = [Int(idxmap[term.variable_2].value) for term in fquadratic.quadratic_terms]
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
