
<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/oxfordcontrol/ClarabelDocs/main/docs/src/assets/logo-banner-dark-jl.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/oxfordcontrol/ClarabelDocs/main/docs/src/assets/logo-banner-light-jl.png">
  <img alt="Clarabel.jl logo" src="https://raw.githubusercontent.com/oxfordcontrol/ClarabelDocs/main/docs/src/assets/logo-banner-light-jl.png" width="66%">
</picture>
<h1 align="center" margin=0px>
GPU implementation of Clarabel solver for Julia
</h1>
   <a href="https://github.com/oxfordcontrol/Clarabel.jl/actions"><img src="https://github.com/oxfordcontrol/Clarabel.jl/workflows/ci/badge.svg?branch=main"></a>
  <a href="https://codecov.io/gh/oxfordcontrol/Clarabel.jl"><img src="https://codecov.io/gh/oxfordcontrol/Clarabel.jl/branch/main/graph/badge.svg"></a>
  <a href="https://oxfordcontrol.github.io/ClarabelDocs/stable"><img src="https://img.shields.io/badge/Documentation-stable-purple.svg"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://github.com/oxfordcontrol/Clarabel.jl/releases"><img src="https://img.shields.io/badge/Release-v0.11.0-blue.svg"></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#license-">License</a> •
  <a href="https://oxfordcontrol.github.io/ClarabelDocs/stable">Documentation</a>
</p>

The branch `CuClarabel` is the GPU implementation of the Clarabel solver, which can solve conic problems of the following form:

$$
\begin{array}{r}
\text{minimize} & \frac{1}{2}x^T P x + q^T x\\\\[2ex]
\text{subject to} & Ax + s = b \\\\[1ex]
        & s \in \mathcal{K}
\end{array}
$$

with decision variables
$x \in \mathbb{R}^n$,
$s \in \mathbb{R}^m$
and data matrices
$P=P^\top \succeq 0$,
$q \in \mathbb{R}^n$,
$A \in \mathbb{R}^{m \times n}$, and
$b \in \mathbb{R}^m$.
The set $\mathcal{K}$ is a composition of convex cones; we support zero cones (linear equality constraints), nonnegative cones (linear inequality constraints), second-order cones, exponential cone, power cones and semidefinite cones of the same dimensionality. Our package relies on the external package [CUDSS.jl](https://github.com/exanauts/CUDSS.jl) for the linear system solver [CUDSS](https://developer.nvidia.com/cudss). We also support linear system solves in lower (mixed) precision.


## Installation
Currently, the GPU implementation is under the branch `CuClarabel` of the package __Clarabel.jl__. You can switch to the GPU version via: `git checkout CuClarabel` under the directory of your local _Clarabel.jl__ package in a terminal. (We aim to merge it back to the `main` branch in the future.)
## Tutorial

### Use in Julia
Modeling a conic optimization problem is the same as in the original [Clarabel solver](https://clarabel.org/stable/), except with the additional parameter `direct_solve_method`. This can be set to `:cudss` or `:cudssmixed`. Here is a portfolio optimization problem modelled via JuMP:
```
using LinearAlgebra, SparseArrays, Random, JuMP
using Clarabel

## generate the data
rng = Random.MersenneTwister(1)
k = 5; # number of factors
n = k * 10; # number of assets
D = spdiagm(0 => rand(rng, n) .* sqrt(k))
F = sprandn(rng, n, k, 0.5); # factor loading matrix
μ = (3 .+ 9. * rand(rng, n)) / 100. # expected returns between 3% - 12%
γ = 1.0; # risk aversion parameter
d = 1 # we are starting from all cash
x0 = zeros(n);

a = 1e-3
b = 1e-1
γ = 1.0;

model = JuMP.Model(Clarabel.Optimizer)
set_optimizer_attribute(model, "direct_solve_method", :cudss)

@variable(model, x[1:n])
@variable(model, y[1:k])   
@variable(model, s[1:n])
@variable(model, t[1:n])
@objective(model, Min, x' * D * x + y' * y - 1/γ * μ' * x);
@constraint(model, y .== F' * x);
@constraint(model, x .>= 0);

# transaction costs
@constraint(model, sum(x) + a * sum(s) == d + sum(x0) );
@constraint(model, [i = 1:n], x0[i] - x[i] == t[i]) 
@constraint(model, [i = 1:n], [s[i], t[i]] in MOI.SecondOrderCone(2));
JuMP.optimize!(model)
```

### Use in Python

We can call julia code within a python file by using [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/) package. We can download the package by
```
pip install juliacall
```
Then, we load the package as a single variable `jl` which represents the Main module in Julia, and we can write Julia code and call it via `jl.seval()` in Python. 
```
from juliacall import Main as jl
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
# Load Clarabel in Julia
jl.seval('using Clarabel, LinearAlgebra, SparseArrays')
jl.seval('using CUDA, CUDA.CUSPARSE')
```
Here we build up a simple optimization problem with a second-order cone, which is fully written by Julia.
```
jl.seval('''
    P = CuSparseMatrixCSR(sparse([2.0 1.0 0.0;
                1.0 2.0 0.0;
                0.0 0.0 2.0]))
    q = CuVector([0, -1., -1])
    A = CuSparseMatrixCSR(SparseMatrixCSC([1. 0 0; -1 0 0; 0 -1 0; 0 0 -1]))
    b = CuVector([1, 0., 0., 0.])

    # 0-cone dimension 1, one second-order-cone of dimension 3
    cones = Dict("f" => 1, "q"=> [3])

    settings = Clarabel.Settings(direct_solve_method = :cudss)
                                    
    solver   = Clarabel.Solver(P,q,A,b,cones, settings)
    Clarabel.solve!(solver)
    
    # Extract solution
    x = solver.solution
''')
```
It is also possible to call the julia functions directly via JuliaCall. For example, if we want to reuse the solver object and update only coefficients in the problem, we can call the following blocks,
```
# Update b vector
bpy = cp.array([2.0, 1.0, 1.0, 1.0], dtype=cp.float64)
bjl = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(bpy.data.ptr), bpy.size)

# "_b" is the replacement of "!" in julia function
jl.Clarabel.update_b_b(jl.solver,bjl)          #Clarabel.update_b!()

# Update P matrix
# Define a new CSR sparse matrix on GPU
Ppy = csr_matrix(cp.array([
    [3.0, 0.5, 0.0],
    [0.5, 2.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=cp.float64))

# Extract the pointers (as integers)
data_ptr    = int(Ppy.data.data.ptr)
indices_ptr = int(Ppy.indices.data.ptr)
indptr_ptr  = int(Ppy.indptr.data.ptr)

# Get matrix shape and non-zero count
n_rows, n_cols = Ppy.shape
nnz = Ppy.nnz

jl.Pjl = jl.Clarabel.cupy_to_cucsrmat(jl.Float64, data_ptr, indices_ptr, indptr_ptr, n_rows, n_cols, nnz)

jl.Clarabel.update_P_b(jl.solver, jl.Pjl)          #Clarabel.update_P!()

#Solve the new problem without creating memory
jl.Clarabel.solve_b(jl.solver)                  #Clarabel.solve!()
```
where we update the linear cost `b` by values in a cupy vector `bpy` and the quadratic cost `P` by values in a cupy csr matrix `Ppy`. Note that we need to replace `!` in a julia function with `_b`. Reversely, we can also extract value from a Julia object back to Python,
```
# Retrieve the solution from Julia to Python
solution = np.array(jl.solver.solution.x)
print("Solution:", solution)
```
The example file can be found under the `python` folder.

### Use in CVXPY
`CuClarabel` is now available in [CVXPY](https://www.cvxpy.org/) as a standalone solver.

### Performance tips
Due to the `just-in-time (JIT)` compilation in Julia, the first call of `CuClarabel` will also be slow in python and it is recommended to solve a mini problem first to trigger the JIT-compilation and get full performance on the subsequent solve of the actual problem. 

## Citing
```
@misc{CuClarabel,
      title={CuClarabel: GPU Acceleration for a Conic Optimization Solver}, 
      author={Yuwen Chen and Danny Tse and Parth Nobel and Paul Goulart and Stephen Boyd},
      year={2024},
      eprint={2412.19027},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2412.19027}, 
}
```

## License 🔍
This project is licensed under the Apache License  2.0 - see the [LICENSE.md](https://github.com/oxfordcontrol/Clarabel.jl/blob/main/LICENSE.md) file for details.

