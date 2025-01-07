module HSLExt

# Load main package and triggers
import Clarabel 
using Clarabel: AbstractDirectLDLSolver, DirectLDLSolversDict
using HSL

export HSLMA57DirectLDLSolver

println("Loading HSLExt...")
include("../src/kktsolvers/direct-ldl/directldl_hsl.jl")

Clarabel.DirectLDLSolversDict[:hsl] = HSLMA57DirectLDLSolver

println("DirectLDLSolversDict keys...", keys(Clarabel.DirectLDLSolversDict))


end

