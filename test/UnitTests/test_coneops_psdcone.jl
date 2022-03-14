using Test, LinearAlgebra, SparseArrays, Clarabel
T = Float64

@testset "coneops_psdcone" begin

    #PJG: These need to go to a common testing utils file

    function randsym(n)
        A = randn(n,n)
        A = A+A'
    end

    function randpsd(n)
        A = randn(n,n)
        A = A*A'
    end

    @testset "test_coneops_psdcone_constructor" begin

        K = Clarabel.PSDCone(25)
        @test Clarabel.dim(K)== 25
        @test Clarabel.degree(K) == 5
        @test_throws DomainError Clarabel.PSDCone(24)
        @test_throws DomainError Clarabel.PSDCone(-1)
        @test_throws DomainError Clarabel.PSDCone(0)

    end

    @testset "test_coneops_psdcone_circle_op" begin

        n = 5
        E = Matrix(I(5)*1.)
        e = E[:]
        X = zeros(n,n)
        Y = randsym(n)
        Z = randsym(n)
        x = X[:]; y = Y[:]; z = Z[:]
        K = Clarabel.PSDCone(n^2)

        Clarabel.circ_op!(K,x,y,z)
        X .= 0.5*(Y*Z + Z*Y)

        @test tr(X) ≈ dot(y,z)
        @test norm(x - X[:]) ≈ 0

    end

    @testset "test_coneops_psdcone_λ_inv_circle_op" begin

        n = 5
        X = randsym(n)
        λdiag = randn(n)
        Λ = Matrix(Diagonal(λdiag))
        Z = zeros(n,n)
        W = zeros(n,n)
        (x,λ,z,w) = map(m->reshape(m,:), (X,Λ,Z,W))
        K = Clarabel.PSDCone(n^2)

        #Z = 1/2(ΛX + XΛ)
        Clarabel.circ_op!(K,z,λ,x)

        #W should now be the solution to 1/2(ΛW + WΛ) = Z
        K.λ .= λdiag        #diagonal internal scaling
        Clarabel.λ_inv_circ_op!(K,w,z)

        #now we should have x = w
        @test norm(x[:]  - w[:]) ≈ 0 atol = 100*eps(T)

    end



    @testset "test_coneops_psdcone_add_scale_e!" begin

        n = 5
        a = randn()
        X = randn(n,n)
        x = X[:];
        K = Clarabel.PSDCone(n^2)
        Clarabel.add_scaled_e!(K,x,a)

        @test norm(reshape(x,n,n) - (X + a*I)) ≈ 0

    end

    @testset "test_coneops_psdcone_shift_to_cone!" begin

        n = 5

        #X is negative definite.   Shift eigenvalues to 1
        X = -randpsd(n)
        x = X[:];
        K = Clarabel.PSDCone(n^2)
        Clarabel.shift_to_cone!(K,x)
        @test minimum(eigvals(reshape(x,n,n))) ≈ 1

        #X is positive definite.   eigenvalues should not change
        X = randpsd(n)
        e = minimum(eigvals(X))
        x = X[:];
        K = Clarabel.PSDCone(n^2)
        Clarabel.shift_to_cone!(K,x)
        @test minimum(eigvals(reshape(x,n,n))) ≈ e

    end


    @testset "test_coneops_psdcone_update_scaling!" begin

        n = 5

        #X is negative definite.   Shift eigenvalues to 1
        S = randpsd(n)
        Z = randpsd(n)

        (s,z) = map(m->reshape(m,:), (S,Z))

        K = Clarabel.PSDCone(n^2)
        Clarabel.update_scaling!(K,s,z)

        f = K.work
        R = f.R
        Rinv = f.Rinv
        Λ = Diagonal(f.λ)

        W = R*R'

        @test norm(W*Z*W' - S) ≈ 0  atol = 1000*eps(T)
        @test norm(R'*Z*R - Λ) ≈ 0  atol = 1000*eps(T)
        @test norm(Rinv*S*Rinv' - Λ) ≈ 0  atol = 1000*eps(T)

    end

    @testset "test_coneops_psdcone_step_length!" begin

        n = 10

        Z = randpsd(n); dZ = randsym(n)
        S = randpsd(n); dS = randsym(n)

        (s,z)   = map(m->reshape(m,:), (S,Z))
        (ds,dz) = map(m->reshape(m,:), (dS,dZ))

        #compute internal scaling required for step calc
        K = Clarabel.PSDCone(n^2)
        Clarabel.update_scaling!(K,s,z)

        #Z direction only
        α = Clarabel.step_length(K,dz,ds.*0.,z,s)
        @test minimum(eigvals(Z + α.*dZ)) ≈ 0.  atol = 1000*eps(T)

        #S direction only
        α = Clarabel.step_length(K,dz.*0,ds,z,s)
        @test minimum(eigvals(S + α.*dS)) ≈ 0.  atol = 1000*eps(T)

        #joint
        α = Clarabel.step_length(K,dz,ds,z,s)
        eS = eigvals(S + α.*dS)
        eZ = eigvals(Z + α.*dZ)
        @test minimum([eS;eZ]) ≈ 0.  atol = 1000*eps(T)

        #unbounded
        dS .= randpsd(n); dZ .= randpsd(n)
        α = Clarabel.step_length(K,dz.*0,ds,z,s)
        @test α ≈ inv(eps(T))  rtol = 10*eps(T)



    end



end
nothing
