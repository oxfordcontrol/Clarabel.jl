using IterativeSolvers, Preconditioners
using SparseArrays, LinearAlgebra

n = 100
m = 200
A = sprand(m,n,0.1)

##############################################
#My iterative solver
##############################################


Pl = Pr = IterativeSolvers.Identity()
for i = 5:8
    # KKT = [spdiagm(0 => 10.0^i*rand(m)) A; A' spdiagm(0 => 10.0^(-i)*rand(n))];
    # b = [ones(n); zeros(m)]
    # x = rand(n+m)

    P = ones(m,n)
    KKT = P*P' + 10.0^(-i)*diagm(0 => ones(m));
    println("cond KKT is: ", cond(KKT))
    b = ones(m)
    x = ones(m)

    p = DiagonalPreconditioner(KKT)
    # p = CholeskyPreconditioner(KKT, 2)

    # println(p)
    # minres!(x, KKT, b)
    cg!(x, KKT, b; Pl = p)
    println(string(i)*"th residual is: ", norm(b - KKT * x) / norm(b))
end
