module Clarabel

    using SparseArrays, LinearAlgebra, QDLDL, AMD, Printf
    const DefaultFloat = Float64
    const DefaultInt   = LinearAlgebra.BlasInt

    #version / release info
    include("./version.jl")

    #core solver components
    include("./consttypes.jl")
    include("./cones/conetypes.jl")
    include("./settings.jl")
    include("./conicvector.jl")
    include("./types.jl")
    include("./variables.jl")
    include("./residuals.jl")
    include("./scalings.jl")
    include("./info.jl")

    #linear subsolver implementations
    #must precede the KKT solver typedef
    include("./linsys/linearsolver_defaults.jl")

    #direct solve methods
    include("./linsys/linearsolver_utils.jl")
    include("./linsys/linearsolver_qdldl.jl")
    include("./linsys/linearsolver_mkl.jl")
    include("./kkt.jl")

    # display, print and top level solver
    include("./printing.jl")
    include("./show.jl")
    include("./solver.jl")

    #conic constraints.  Additional
    #cone implementations here
    include("./cones/coneops.jl")
    include("./cones/coneops_defaults.jl")
    include("./cones/coneops_zerocone.jl")
    include("./cones/coneops_nncone.jl")
    include("./cones/coneops_socone.jl")
    include("./cones/coneops_psdcone.jl")

    #equilibration and various algebraic
    #utilities
    include("./equilibration.jl")
    include("./mathutils.jl")

    #MathOptInterface for JuMP/Convex.jl
    include("./MOI_wrapper/MOI_wrapper.jl")

end
