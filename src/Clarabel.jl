module Clarabel

    using SparseArrays, LinearAlgebra, Printf
    const DefaultFloat = Float64
    const DefaultInt   = LinearAlgebra.BlasInt
    const IdentityMatrix = UniformScaling{Bool}


    #version / release info
    include("./version.jl")

    #core solver components
    include("./cones/conetypes.jl")
    include("./cones/coneset.jl")
    include("./settings.jl")
    include("./conicvector.jl")
    include("./types.jl")
    include("./variables.jl")
    include("./residuals.jl")
    include("./equilibration.jl")
    include("./info.jl")
    include("./result.jl")

    #direct LDL linear solve methods
    include("./kktsolvers/direct-ldl/includes.jl")

    #KKT solvers and solver level kktsystem
    include("./kktsolvers/kktsolver_defaults.jl")
    include("./kktsolvers/kktsolver_directldl.jl")
    include("./kktsystem.jl")

    # printing and top level solver
    include("./printing.jl")
    include("./solver.jl")

    #conic constraints.  Additional
    #cone implementations go here
    include("./cones/coneops.jl")
    include("./cones/coneops_defaults.jl")
    include("./cones/coneops_zerocone.jl")
    include("./cones/coneops_nncone.jl")
    include("./cones/coneops_socone.jl")
    include("./cones/coneops_psdtrianglecone.jl")

    #various algebraic utilities
    include("./utils/mathutils.jl")
    include("./utils/csc_assembly.jl")

    #MathOptInterface for JuMP/Convex.jl
    include("./MOI_wrapper/MOI_wrapper.jl")

end
