
include("./directldl_defaults.jl")
include("./directldl_qdldl.jl")
include("./directldl_mklpardiso.jl")
include("./directldl_cholmod.jl")
include("./directldl_utils.jl")

#mapping of direct LDL solver type symbol
#from user settings to corresponding subtype
#of AbstractDirectLDLSolver to use.

const DirectLDLSolversDict = Dict(
    :qdldl    =>  QDLDLDirectLDLSolver,
    :mkl      =>  PardisoDirectLDLSolver,
    :cholmod  =>  CholmodDirectLDLSolver,
)
