module IPSolver

    using SparseArrays, LinearAlgebra, QDLDL, AMD, Printf
    const DefaultFloat = Float64
    const DefaultInt   = LinearAlgebra.BlasInt

    include("./consttypes.jl")
    include("./settings.jl")
    include("./cones.jl")
    include("./types.jl")
    include("./variables.jl")
    include("./scalings.jl")
    include("./kktsolver_direct.jl")
    include("./kktsolver_indirect_minres.jl")
    include("./printing.jl")
    include("./coneops.jl")
    include("./solver.jl")

end
