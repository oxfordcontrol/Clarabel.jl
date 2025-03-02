module PardisoExt

# Load main package and triggers
using Clarabel, Pardiso

include("../src/kktsolvers/direct-ldl/directldl_pardiso.jl")

end