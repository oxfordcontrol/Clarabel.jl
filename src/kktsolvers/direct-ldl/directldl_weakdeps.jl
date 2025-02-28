const DirectLDLSolversPackageDict = Dict{Symbol, String}()

# ClarabelHSL
abstract type HSLDirectLDLSolver{T} <: AbstractDirectLDLSolver{T} end
DirectLDLSolversDict[:ma57] = HSLDirectLDLSolver
DirectLDLSolversPackageDict[:ma57] = "HSL"

# ClarabelPardiso
abstract type AbstractPardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}  end
abstract type AbstractMKLPardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}  end
DirectLDLSolversDict[:mkl] = AbstractMKLPardisoDirectLDLSolver
DirectLDLSolversPackageDict[:mkl] = "Pardiso"
abstract type AbstractPanuaPardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}  end
DirectLDLSolversDict[:panua] = AbstractPanuaPardisoDirectLDLSolver
DirectLDLSolversPackageDict[:panua] = "Pardiso"

# Generic fallback
function _is_ldlsolver_implemented(ldlsolver, s::Symbol) 
    throw(error("Using direct LDL solver :", s, " requires \"using ", DirectLDLSolversPackageDict[s], "\""))
end
