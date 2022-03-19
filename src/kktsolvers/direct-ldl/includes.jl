
include("./directldl_defaults.jl")
include("./directldl_qdldl.jl")
include("./directldl_mkl.jl")
include("./directldl_utils.jl")

#mapping of direct LDL solver type symbol
#from user settings to corresponding subtype
#of AbstractDirectLDLSolver to use.

const DirectLDLSolversDict = Dict(
    :qdldl    =>  QDLDLDirectLDLSolver,
    :mkl      =>  PardisoDirectLDLSolver,
)
