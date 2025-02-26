const DirectLDLSolversPackageDict = Dict{Symbol, String}()

abstract type HSLDirectLDLSolver{T} <: AbstractDirectLDLSolver{T} end
DirectLDLSolversDict[:ma57] = HSLDirectLDLSolver
DirectLDLSolversPackageDict[:ma57] = "HSL"

abstract type AbstractMKLPardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}  end
DirectLDLSolversDict[:mkl] = AbstractMKLPardisoDirectLDLSolver
DirectLDLSolversPackageDict[:mkl] = "Pardiso"

abstract type AbstractPanuaPardisoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}  end
DirectLDLSolversDict[:panua] = AbstractPanuaPardisoDirectLDLSolver
DirectLDLSolversPackageDict[:panua] = "Pardiso"

function _is_ldlsolver_implemented(ldlsolver, s::Symbol) 
    throw(error("Using direct LDL solver :", s, " requires \"using ", DirectLDLSolversPackageDict[s], "\""))
end
