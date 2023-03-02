__precompile__()
module Clarabel

    #internal constraint RHS limits 
    const INFINITY = Ref{Float64}(1e20)
    set_infinity(v::Float64) = Clarabel.INFINITY[] =  v
    get_infinity() = Clarabel.INFINITY[]

    using SparseArrays, LinearAlgebra, Printf, Requires
    const DefaultFloat = Float64
    const DefaultInt   = LinearAlgebra.BlasInt
    const IdentityMatrix = UniformScaling{Bool}

    #version / release info
    include("./version.jl")

    #API for user cone specifications
    include("./cones/cone_api.jl")

    #cone type definitions
    include("./cones/cone_types.jl")
    include("./cones/compositecone_type.jl")

    #core solver components
    include("./settings.jl")
    include("./conicvector.jl")
    include("./statuscodes.jl")
    include("./presolver.jl")
    include("./types.jl")
    include("./variables.jl")
    include("./residuals.jl")
    include("./equilibration.jl")
    include("./info.jl")
    include("./solution.jl")

    #direct LDL linear solve methods
    include("./kktsolvers/direct-ldl/includes.jl")

    #KKT solvers and solver level kktsystem
    include("./kktsolvers/kktsolver_defaults.jl")
    include("./kktsolvers/kktsolver_directldl.jl")
    include("./kktsystem.jl")

    # printing and top level solver
    include("./info_print.jl")
    include("./solver.jl")

    #conic constraints.  Additional
    #cone implementations go here
    include("./cones/coneops_defaults.jl")
    include("./cones/coneops_zerocone.jl")
    include("./cones/coneops_nncone.jl")
    include("./cones/coneops_socone.jl")
    include("./cones/coneops_psdtrianglecone.jl")
    include("./cones/coneops_expcone.jl")
    include("./cones/coneops_powcone.jl")
    include("./cones/coneops_compositecone.jl")
    include("./cones/coneops_exppow_common.jl")
    include("./cones/coneops_symmetric_common.jl")

    #various algebraic utilities
    include("./utils/mathutils.jl")
    include("./utils/csc_assembly.jl")

    #optional dependencies.  
    #NB: This __init__ function and its @require statements 
    #should be removed upon update of this package for use 
    #with Julia v1.10+, after which weakdeps / external 
    #dependencies will be natively supported 
    function __init__()
        @require Pardiso="46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2" begin
            include("./kktsolvers/direct-ldl/directldl_mklpardiso.jl")  
        end 
    end

    #MathOptInterface for JuMP/Convex.jl
    module MOImodule
         include("./MOI_wrapper/MOI_wrapper.jl")
    end
    const Optimizer{T} = Clarabel.MOImodule.Optimizer{T}

end

