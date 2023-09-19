using MathOptInterface, SparseArrays
using ..Clarabel
export Optimizer

#-----------------------------
# Const definitions
#-----------------------------

const MOI = MathOptInterface
const MOIU = MOI.Utilities
const SparseTriplet{T} = Tuple{Vector{<:Integer}, Vector{<:Integer}, Vector{T}}

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
    solver_nvars::Union{Nothing,Int}
    use_quad_obj::Bool
    sense::MOI.OptimizationSense
    objconstant::T
    rowranges::Dict{Int, UnitRange{Int}}

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
        rowranges = Dict{Int, UnitRange{Int}}()
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
    optimizer.rowranges = Dict{Int, UnitRange{Int}}()
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
        solvetime = round.(optimizer.solver_info.solve_time*1000,digits=2)
        println(io, " : Solve time: $(solvetime)ms")
        end
    end
end


#-----------------------------
# Solver Attributes, get/set
#-----------------------------

MOI.get(opt::Optimizer, ::MOI.SolverName)        = string(opt.solver_module)
MOI.get(opt::Optimizer, ::MOI.SolverVersion)     = Clarabel.version()
MOI.get(opt::Optimizer, ::MOI.RawSolver)         = opt.solver
MOI.get(opt::Optimizer, ::MOI.ResultCount)       = Int(!isnothing(opt.solver_solution))
MOI.get(opt::Optimizer, ::MOI.NumberOfVariables) = opt.solver_nvars
MOI.get(opt::Optimizer, ::MOI.SolveTimeSec)      = opt.solver_info.solve_time
MOI.get(opt::Optimizer, ::MOI.RawStatusString)   = string(opt.solver_info.status)
MOI.get(opt::Optimizer, ::MOI.BarrierIterations) = Int(opt.solver_info.iterations)

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


#attributes not currently supported
MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = false


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

    idxmap = MOIU.IndexMap(dest, src)

    #check all model/variable/constraint attributes to
    #ensure that everything passed is handled by the solver
    copy_to_check_attributes(dest,src)

    #assemble the constraints data
    assign_constraint_row_ranges!(dest.rowranges, idxmap, src)
    A, b, cone_spec, = process_constraints(dest, src, idxmap)

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
    b = Vector{T}(undef, m)

    #these will be used for a triplet representation of A
    I = Int[]
    J = Int[]
    V = T[]

    #these will be used for the Clarabel API cone types
    cone_spec = Clarabel.SupportedCone[]

    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        push_constraint!(
            (I, J, V), b, cone_spec,
            src, idxmap, rowranges, F, S)
    end

    #we have built Ax + b \in Cone, but we actually
    #want to pose the problem as Ax + s = b, s\ in Cone
    V .= -V  #changes sign of A

    n = MOI.get(src, MOI.NumberOfVariables())
    A = sparse(I, J, V, m, n)

    return (A, b, cone_spec)

end

function push_constraint!(
    triplet::SparseTriplet,
    b::Vector{T},
    cone_spec::Vector{Clarabel.SupportedCone},
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
        push_constraint_constant!(b, rows, f, s)
        push_constraint_linear!(triplet, f, rows, idxmap, s)
        push_constraint_set!(cone_spec, rows, s)
    end

    return nothing
end

function push_constraint_constant!(
    b::AbstractVector{T},
    rows::UnitRange{Int},
    f::MOI.VectorAffineFunction{T},
    ::OptimizerSupportedMOICones{T},
) where {T}

    b[rows] .= f.constants
    return nothing
end

function push_constraint_constant!(
    b::AbstractVector{T},
    rows::UnitRange{Int},
    f::MOI.VectorOfVariables,
    s::OptimizerSupportedMOICones{T},
) where {T}
    b[rows] .= zero(T)
    return nothing
end

function push_constraint_linear!(
    triplet::SparseTriplet,
    f::MOI.VectorAffineFunction{T},
    rows::UnitRange{Int},
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
    rows::UnitRange{Int},
    idxmap,
    s::OptimizerSupportedMOICones{T},
) where {T}

    (I, J, V) = triplet
    cols = [idxmap[var].value for var in f.variables]
    append!(I, rows)
    append!(J, cols)
    vals = ones(T, length(cols))
    append!(V, vals)

    return nothing
end


function push_constraint_set!(
    cone_spec::Vector{Clarabel.SupportedCone},
    rows::Union{Int,UnitRange{Int}},
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
        pow_cone_type = MOItoClarabelCones[MOI.ExponentialCone]
        push!(cone_spec, pow_cone_type())
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
            P = spzeros(T, n, n)
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
