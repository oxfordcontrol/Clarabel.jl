using SuiteSparse

mutable struct CholmodDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    KKT::SparseMatrixCSC{T}
    KKTsym::Symmetric{T, SparseMatrixCSC{T,Int}}
    F::Union{SuiteSparse.CHOLMOD.Factor,Nothing}

    function CholmodDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        #NB: ignore Dsigns here because Cholmod doesn't
        #use information about the expected signs

        #cholmod requires that we always pass it a symmetric
        #view, so we carry KKTsym for cholmod but modify KKT
        KKTsym = Symmetric(KKT)

        #There is no obvious way to force cholmod to make
        #an initial symbolic factorization only, set
        #set F to nothing and create F as needed on the
        #first refactorization
        F = nothing

        return new(KKT,KKTsym,F)
    end
end

DirectLDLSolversDict[:cholmod] = CholmodDirectLDLSolver
required_matrix_shape(::Type{CholmodDirectLDLSolver}) = :triu

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::CholmodDirectLDLSolver{T},
    index::Vector{Int},
    values::Vector{T}
) where{T}

    @views ldlsolver.KKT.nzval[index] .= values
end

#offset entries in the KKT matrix using the
#given index into its CSC representation and
#an optional vector of signs
function offset_values!(
    ldlsolver::CholmodDirectLDLSolver{T},
    index::Vector{Int},
    offset::Union{T,Vector{T}},
    signs::Union{Int,Vector{Int}} = 1
) where{T}

    @views @. ldlsolver.KKT.nzval[index] += offset*signs
end


#refactor the linear system
function refactor!(ldlsolver::CholmodDirectLDLSolver{T}) where{T}

    if ldlsolver.F === nothing
        #initial symbolic and numeric factor since
        #we can't do symbolic on its own
        ldlsolver.F = ldlt(ldlsolver.KKTsym)

    else
        #this reuses the symbolic factorization
        ldlt!(ldlsolver.F, ldlsolver.KKTsym)
    end
end


#solve the linear system
function solve!(
    ldlsolver::CholmodDirectLDLSolver{T},
    x::Vector{T},
    b::Vector{T},
    settings
) where{T}

    x .= ldlsolver.F\b
end
