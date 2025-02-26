module ClarabelPardiso

using SparseArrays
import Clarabel
import Pardiso

# Shared functionalities
function pardiso_init(ps,KKT,Dsigns,settings)

    #NB: ignore Dsigns here because pardiso doesn't
    #use information about the expected signs

    #perform logical factor
    Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
    Pardiso.pardisoinit(ps)
    Pardiso.fix_iparm!(ps, :N)
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
    Pardiso.pardiso(ps, KKT, [1.])  #RHS irrelevant for ANALYSIS
end 

# MKL Pardiso variant
struct MKLPardisoDirectLDLSolver{T} <: Clarabel.AbstractDirectLDLSolver{T}
    ps::Pardiso.MKLPardisoSolver

    function MKLPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}
        Pardiso.mkl_is_available() || error("MKL Pardiso is not available")
        ps = Pardiso.MKLPardisoSolver()
        pardiso_init(ps,KKT,Dsigns,settings)
        return new(ps)
    end
end

Clarabel.required_matrix_shape(::Type{MKLPardisoDirectLDLSolver}) = :tril
Clarabel._is_ldlsolver_implemented(ldlsolver::Type{Clarabel.AbstractMKLPardisoDirectLDLSolver}, s::Symbol) = MKLPardisoDirectLDLSolver

#update entries in the KKT matrix using the
#given index into its CSC representation
function Clarabel.update_values!(
    ldlsolver::MKLPardisoDirectLDLSolver{T},
    index::AbstractVector{Clarabel.DefaultInt},
    values::Vector{T}
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!
end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function Clarabel.scale_values!(
    ldlsolver::MKLPardisoDirectLDLSolver{T},
    index::AbstractVector{Clarabel.DefaultInt},
    scale::T
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!
end


#refactor the linear system
function Clarabel.refactor!(ldlsolver::MKLPardisoDirectLDLSolver{T},K::SparseMatrixCSC{T}) where{T}

    # Pardiso is quite robust and will usually produce some 
    # kind of factorization unless there is an explicit 
    # zero pivot or some other nastiness.   "success" 
    # here just means that it didn't fail outright, although 
    # the factorization could still be garbage 

    # Recompute the numeric factorization susing fake RHS
    try 
        Pardiso.set_phase!(ldlsolver.ps, Pardiso.NUM_FACT)
        Pardiso.pardiso(ldlsolver.ps, K, [1.])
        return is_success = true
    catch 
        return is_success = false
    end
     
end


#solve the linear system
function Clarabel.solve!(
    ldlsolver::MKLPardisoDirectLDLSolver{T},
    KKT::SparseMatrixCSC{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    ps  = ldlsolver.ps
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, x, KKT, b)

end


# Panua Pardiso variant
struct PanuaPardisoDirectLDLSolver{T} <: Clarabel.AbstractDirectLDLSolver{T}
    ps::Pardiso.PardisoSolver

    function PanuaPardisoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        Pardiso.panua_is_available() || error("Panua Pardiso is not available")
        ps = Pardiso.PardisoSolver()
        pardiso_init(ps,KKT,Dsigns,settings)
        ps.iparm[8]=-99 # No IR
        return new(ps)
    end
end

Clarabel.required_matrix_shape(::Type{PanuaPardisoDirectLDLSolver}) = :tril
Clarabel._is_ldlsolver_implemented(ldlsolver::Type{Clarabel.AbstractPanuaPardisoDirectLDLSolver}, s::Symbol) = PanuaPardisoDirectLDLSolver

#update entries in the KKT matrix using the
#given index into its CSC representation
function Clarabel.update_values!(
    ldlsolver::PanuaPardisoDirectLDLSolver{T},
    index::AbstractVector{Clarabel.DefaultInt},
    values::Vector{T}
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!
end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function Clarabel.scale_values!(
    ldlsolver::PanuaPardisoDirectLDLSolver{T},
    index::AbstractVector{Clarabel.DefaultInt},
    scale::T
) where{T}

    #no-op.  Will just use KKT matrix as it as
    #passed to refactor!
end


#refactor the linear system
function Clarabel.refactor!(ldlsolver::PanuaPardisoDirectLDLSolver{T},K::SparseMatrixCSC{T}) where{T}

    # Pardiso is quite robust and will usually produce some 
    # kind of factorization unless there is an explicit 
    # zero pivot or some other nastiness.   "success" 
    # here just means that it didn't fail outright, although 
    # the factorization could still be garbage 

    # Recompute the numeric factorization susing fake RHS
    try 
        Pardiso.set_phase!(ldlsolver.ps, Pardiso.NUM_FACT)
        Pardiso.pardiso(ldlsolver.ps, K, [1.])
        return is_success = true
    catch 
        return is_success = false
    end
     
end


#solve the linear system
function Clarabel.solve!(
    ldlsolver::PanuaPardisoDirectLDLSolver{T},
    KKT::SparseMatrixCSC{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    ps  = ldlsolver.ps
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, x, KKT, b)

end

end