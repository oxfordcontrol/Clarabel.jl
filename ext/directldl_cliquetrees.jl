using CliqueTrees.Multifrontal, SparseArrays, LinearAlgebra, Clarabel
using CliqueTrees.Multifrontal: FVector, flatindices, setflatindex!

import Clarabel: DefaultInt, AbstractDirectLDLSolver, LinearSolverInfo
import Clarabel: ldlsolver_constructor, ldlsolver_matrix_shape, ldlsolver_is_available
import Clarabel: linear_solver_info, update_values!, scale_values!, refactor!, solve!

struct CliqueTreesDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}
    F::ChordalLDLt{:L, T, Int, FVector{T}, FVector{Int}}
    P::Vector{Int}
    reg::DynamicRegularization{T}
    signs::Vector{Int}

    function CliqueTreesDirectLDLSolver{T}(KKT::SparseMatrixCSC{T}, signs, settings) where {T}
        A = Symmetric(KKT, :L)
        F = ChordalLDLt{:L}(A)
        P = flatindices(F, A)

        reg = DynamicRegularization(;
            delta=convert(T, settings.dynamic_regularization_delta),
            epsilon=convert(T, settings.dynamic_regularization_eps),
        )

        return new{T}(F, P, reg, signs)
    end
end

ldlsolver_constructor(::Val{:cliquetrees}) = CliqueTreesDirectLDLSolver
ldlsolver_matrix_shape(::Val{:cliquetrees}) = :tril
ldlsolver_is_available(::Val{:cliquetrees}) = true

function linear_solver_info(solver::CliqueTreesDirectLDLSolver{T}) where {T}
    name = :cliquetrees
    threads = 1
    direct = true
    nnzA = length(solver.P)
    nnzL = nnz(solver.F)
    LinearSolverInfo(name, threads, direct, nnzA, nnzL)
end

function update_values!(
    solver::CliqueTreesDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
    values::Vector{T}
) where {T}
    return
end

function scale_values!(
    solver::CliqueTreesDirectLDLSolver{T},
    index::AbstractVector{DefaultInt},
    scale::T
) where {T}
    return
end

function refactor!(solver::CliqueTreesDirectLDLSolver{T}, K::SparseMatrixCSC{T}) where {T}
    F = solver.F
    P = solver.P
    reg = solver.reg
    signs = solver.signs
    nzval = K.nzval

    fill!(F, zero(T))

    @inbounds for i in eachindex(P)
        setflatindex!(F, nzval[i], P[i])
    end

    ldlt!(F; check=false, reg, signs)
    return issuccess(F)
end

function solve!(
    solver::CliqueTreesDirectLDLSolver{T},
    K::SparseMatrixCSC{T},
    x::Vector{T},
    b::Vector{T}
) where {T}
    ldiv!(x, solver.F, b)
    return
end
