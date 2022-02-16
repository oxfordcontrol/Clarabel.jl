module IPSolver

    using SparseArrays, LinearAlgebra, QDLDL, AMD, Printf
    const DefaultFloat = Float64
    const DefaultInt   = LinearAlgebra.BlasInt

    include("./consttypes.jl")
    include("./cones/conetypes.jl")
    include("./settings.jl")

    include("./types.jl")
    include("./variables.jl")
    include("./residuals.jl")
    include("./scalings.jl")
    include("./info.jl")

    #linear subsolver implementations
    #must precede the KKT solver typedef
    include("./linsys/linearsolver_defaults.jl")
    include("./linsys/linearsolver_qdldl.jl")
    include("./linsys/linearsolver_utils.jl")

    include("./kkt.jl")
    include("./printing.jl")
    include("./solver.jl")

    include("./cones/coneops.jl")
    include("./cones/coneops_defaults.jl")
    include("./cones/coneops_zerocone.jl")
    include("./cones/coneops_nncone.jl")
    include("./cones/coneops_socone.jl")

    #PJG : temporary debugging utils
    include("./debug.jl")
    include("./debug_coneops.jl")
    #include("./kkt_debug.jl")

end
