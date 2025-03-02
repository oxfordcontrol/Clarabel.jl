
include("./directldl_defaults.jl")
include("./directldl_qdldl.jl")
include("./directldl_cholmod.jl")
include("./directldl_kkt_assembly.jl")
include("./directldl_datamaps.jl")

#NB: HSL and Pardiso are weakdeps and are not included here.  
#Loading is done from <packageroot>/ext 

# wrappers to allow calls directly on symbols 
ldlsolver_matrix_shape(x::Symbol) = ldlsolver_matrix_shape(Val{x}())
ldlsolver_constructor(x::Symbol) = ldlsolver_constructor(Val{x}())
