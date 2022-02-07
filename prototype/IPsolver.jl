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
    include("./status.jl")
    include("./kktsolver_direct.jl")
    include("./kktsolver_indirect.jl")
    include("./printing.jl")
    include("./solver.jl")

    include("./cones/coneops.jl")
    include("./cones/coneops_defaults.jl")
    include("./cones/coneops_zerocone.jl")
    include("./cones/coneops_nncone.jl")
    include("./cones/coneops_socone.jl")

    #PJG : temporary debugging utils
    include("./debug.jl")

end
