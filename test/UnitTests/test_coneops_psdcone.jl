using Test, LinearAlgebra, SparseArrays, Clarabel, Random
include("../testing_utils.jl")

rng = Random.MersenneTwister(242713)

FloatT = Float64

@testset "coneops_psdcone" begin

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
        Y = randsym(rng, n)
        Z = randsym(rng, n)
        x = X[:]; y = Y[:]; z = Z[:]
        K = Clarabel.PSDCone(n^2)

        Clarabel.circ_op!(K,x,y,z)
        X .= 0.5*(Y*Z + Z*Y)

        @test tr(X) ≈ dot(y,z)
        @test norm(x - X[:]) ≈ 0

    end

    @testset "test_coneops_psdcone_λ_inv_circle_op" begin

        n = 5
        X = randsym(rng,n)
        λdiag = randn(rng,n)
        Λ = Matrix(Diagonal(λdiag))
        Z = zeros(n,n)
        W = zeros(n,n)
        (x,λ,z,w) = map(m->reshape(m,:), (X,Λ,Z,W))
        K = Clarabel.PSDCone(n^2)

        #Z = 1/2(ΛX + XΛ)
        Clarabel.circ_op!(K,z,λ,x)

        #W should now be the solution to 1/2(ΛW + WΛ) = Z
        K.work.λ .= λdiag        #diagonal internal scaling
        Clarabel.λ_inv_circ_op!(K,w,z)

        #now we should have x = w
        @test norm(x[:]  - w[:]) ≈ 0 atol = 100*eps(FloatT)

    end



    @testset "test_coneops_psdcone_add_scale_e!" begin

        n = 5
        a = 0.12345
        X = randsym(rng,n)
        x = X[:];
        K = Clarabel.PSDCone(n^2)
        Clarabel.add_scaled_e!(K,x,a)

        @test norm(reshape(x,n,n) - (X + a*I)) ≈ 0

    end

    @testset "test_coneops_psdcone_shift_to_cone!" begin

        n = 5

        #X is negative definite.   Shift eigenvalues to 1
        X = -randpsd(rng,n)
        x = X[:];
        K = Clarabel.PSDCone(n^2)
        Clarabel.shift_to_cone!(K,x)
        @test minimum(eigvals(reshape(x,n,n))) ≈ 1

        #X is positive definite.   eigenvalues should not change
        X = randpsd(rng,n)
        e = minimum(eigvals(X))
        x = X[:];
        K = Clarabel.PSDCone(n^2)
        Clarabel.shift_to_cone!(K,x)
        @test minimum(eigvals(reshape(x,n,n))) ≈ e

    end


    @testset "test_coneops_psdcone_update_scaling!" begin

        n = 5

        #X is negative definite.   Shift eigenvalues to 1
        S = randpsd(rng,n)
        Z = randpsd(rng,n)

        (s,z) = map(m->reshape(m,:), (S,Z))

        K = Clarabel.PSDCone(n^2)
        Clarabel.update_scaling!(K,s,z)

        f = K.work
        R = f.R
        Rinv = f.Rinv
        Λ = Diagonal(f.λ)

        W = R*R'

        @test norm(W*Z*W' - S) ≈ 0  atol = 1000*eps(FloatT)
        @test norm(R'*Z*R - Λ) ≈ 0  atol = 1000*eps(FloatT)
        @test norm(Rinv*S*Rinv' - Λ) ≈ 0  atol = 1000*eps(FloatT)

    end

    @testset "test_coneops_psdcone_step_length!" begin

        n = 10

        Z = randpsd(rng,n); dZ = randsym(rng,n)
        S = randpsd(rng,n); dS = randsym(rng,n)

        (s,z)   = map(m->reshape(m,:), (S,Z))
        (ds,dz) = map(m->reshape(m,:), (dS,dZ))

        #compute internal scaling required for step calc
        K = Clarabel.PSDCone(n^2)
        Clarabel.update_scaling!(K,s,z)

        #Z direction only
        α = Clarabel.step_length(K,dz,ds.*0.,z,s)[1]
        @test minimum(eigvals(Z + α.*dZ)) ≈ 0.  atol = 1000*eps(FloatT)

        #S direction only
        α = Clarabel.step_length(K,dz.*0,ds,z,s)[2]
        @test minimum(eigvals(S + α.*dS)) ≈ 0.  atol = 1000*eps(FloatT)

        #joint
        (αz,αs) = Clarabel.step_length(K,dz,ds,z,s)
        eZ = eigvals(Z + αz.*dZ)
        eS = eigvals(S + αs.*dS)
        @test minimum(eZ) ≈ 0.  atol = 1000*eps(FloatT)
        @test minimum(eS) ≈ 0.  atol = 1000*eps(FloatT)

        #unbounded
        dS .= randpsd(rng,n); dZ .= randpsd(rng,n)
        (αz,αs) = Clarabel.step_length(K,dz.*0,ds,z,s)
        @test min(αz,αs) ≈ inv(eps(FloatT))  rtol = 10*eps(FloatT)

    end

    @testset "test_coneops_psdcone_WtW_operations!" begin

        n = 5

        (Z,S) = map(m->randpsd(rng,n), 1:2)
        (V1,V2,V3) = map(m->randpsd(rng,n), 1:3)
        (s,z,v1,v2,v3) = map(m->reshape(m,:), (S,Z,V1,V2,V3))

        #compute internal scaling required for step calc
        K = Clarabel.PSDCone(n^2)
        Clarabel.update_scaling!(K,s,z)

        R    = K.work.R
        Rinv = K.work.Rinv

        #compare different ways of multiplying v by W and W^T
        # v2 = W*v1
        Clarabel.gemv_W!(K,:N,v1,v2,one(FloatT),zero(FloatT))
        # v3 = W^T*v2
        Clarabel.gemv_W!(K,:T,v2,v3,one(FloatT),zero(FloatT))

        WtW = triu(ones(n^2,n^2))       #s is n^2 long
        idxWtW = findall(WtW .!= 0)
        vecWtW = zeros(FloatT,length(idxWtW))
        Clarabel.get_WtW_block!(K,vecWtW)
        WtW[idxWtW] = vecWtW
        #make Symmetric for products
        WtWsym = Symmetric(WtW)

        @test norm(WtWsym*v1 - v3) ≈ 0   atol = 1e-10
        #now the inverse
        Clarabel.gemv_Winv!(K,:T,v1,v2,one(FloatT),zero(FloatT))
        # v3 = W^T*v2
        Clarabel.gemv_Winv!(K,:N,v2,v3,one(FloatT),zero(FloatT))
        @test norm(WtWsym\v1 - v3) ≈ 0   atol = 1e-10

    end



end
nothing
