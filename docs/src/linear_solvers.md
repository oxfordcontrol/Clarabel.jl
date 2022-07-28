# Linear System Solvers

The primary numerical operation inside Clarabel.jl is the solution of a symmetric quasidefinite linear system at each iteration.  The solver currently supports three different solvers to perform factorization and forward/backward substitution on this system.

The linear solver can be configured in Settings using the `direct_solve_method` field, e.g.

```julia
settings = Solver.Settings(direct_solve_method = :qdldl)
```

The solvers currently supported are

Symbol | Package | Description
---  | :--- | :---
:qdldl | [QDLDL.jl](https://github.com/osqp/QDLDL.jl)   | Default solver
:mkl   | [Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl) | Intel MKL Pardiso
:cholmod | Julia native [ldlt](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.ldlt) solver | SuiteSparse.CHOLMOD

!!! note
    To use the MKL Pardiso solver you must install the respective libraries and the corresponding Julia wrapper. For more information about installing these, visit the [Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl) repository page.


QDLDL is a single threaded solver written in pure Julia, and is generally adequate for problems of small to medium size.   The MKL Pardiso solver is multi-threaded and may be preferred for very large problem instances, or problems in which the problem data is extremely dense.

Support for additional linear system solvers may be implemented in future releases.   
