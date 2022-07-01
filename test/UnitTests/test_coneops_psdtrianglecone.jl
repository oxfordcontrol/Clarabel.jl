using Test, LinearAlgebra, SparseArrays, Clarabel, Random
include("../testing_utils.jl")

rng = Random.MersenneTwister(242713)

FloatT = Float64

@testset "coneops_PSDTriangleCone" begin

    @testset "test_coneops_psdtrianglecone_constructor" begin

        K = Clarabel.PSDTriangleCone(5)
        @test Clarabel.numel(K)== 15
        @test Clarabel.degree(K) == 5
        @test_throws DomainError Clarabel.PSDTriangleCone(-1)
        @test_throws DomainError Clarabel.PSDTriangleCone(0)

    end

    @testset "test_coneops_psdtrianglecone_conversions" begin

        n = 5
        K = Clarabel.PSDTriangleCone(n)
        X = randsym(rng, n)
        Y = randsym(rng, n)
        Z = zeros(n,n)
        x = zeros((n*(n+1))>>1)
        y = zeros((n*(n+1))>>1)

        map((v,M)->Clarabel._mat_to_svec!(v,M,K), (x,y), (X,Y))
        @test x'y - tr(X'Y)≈ 0  atol = 1e-12

        Clarabel._mat_to_svec!(x,X,K)
        Clarabel._svec_to_mat!(Z,x,K)
        @test norm(X-Z) ≈ 0     atol = 1e-12


    end

    @testset "test_coneops_psdtrianglecone_circle_op" begin

        n = 5
        K = Clarabel.PSDTriangleCone(n)
        X1 = zeros(n,n)
        X2 = zeros(n,n)
        Y = randsym(rng, n)
        Z = randsym(rng, n)
        (x,y,z) = map(v->zeros((n*(n+1))>>1),1:3)
        map((v,M)->Clarabel._mat_to_svec!(v,M,K), (y,z), (Y,Z))
        K = Clarabel.PSDTriangleCone(n)

        X1 .= 0.5*(Y*Z + Z*Y)

        Clarabel.circ_op!(K,x,y,z)
        Clarabel._svec_to_mat!(X2,x,K)

        @test tr(X1) ≈ dot(y,z)
        @test norm(X2-X1) ≈ 0   atol = 1e-12

    end


    @testset "test_coneops_psdtrianglecone_λ_inv_circle_op" begin

        n = 5
        K = Clarabel.PSDTriangleCone(n)
        X     = randsym(rng,n)
        λdiag = randn(rng,n)
        Λ = Matrix(Diagonal(λdiag))
        Z = zeros(n,n)
        W = zeros(n,n)
        (x,z,λ,w) = map(m->zeros(K.numel), 1:4)
        map((v,M)->Clarabel._mat_to_svec!(v,M,K),(x,z,λ,w),(X,Z,Λ,W))

        #Z = 1/2(ΛX + XΛ)
        Clarabel.circ_op!(K,z,λ,x)

        #W should now be the solution to 1/2(ΛW + WΛ) = Z
        K.work.λ .= λdiag        #diagonal internal scaling
        Clarabel.λ_inv_circ_op!(K,w,z)
        Clarabel._svec_to_mat!(W,w,K)

        #now we should have x = w
        @test norm(x  - w) ≈ 0 atol = 100*eps(FloatT)
        #now we should have x = w
        @test norm(X  - W) ≈ 0 atol = 100*eps(FloatT)

    end



    @testset "test_coneops_psdtrianglecone_add_scale_e!" begin

        n = 5
        a = 0.12345
        K = Clarabel.PSDTriangleCone(n)
        X = randsym(rng,n)
        X .= 1. * 0.
        x = zeros(K.numel)

        XplusaI = X + a*I(n)
        Clarabel._mat_to_svec!(x,X,K)
        Clarabel.add_scaled_e!(K,x,a)
        Clarabel._svec_to_mat!(X,x,K)

        @test norm(X - XplusaI) ≈ 0

    end

    @testset "test_coneops_psdtrianglecone_shift_to_cone!" begin

        n = 5
        K = Clarabel.PSDTriangleCone(n)

        #X is negative definite.   Shift eigenvalues to 1
        X = -randpsd(rng,n).*0
        X = X - 1e-10*I(n)
        x = zeros(K.numel)
        Clarabel._mat_to_svec!(x,X,K)
        Clarabel.shift_to_cone!(K,x)
        Clarabel._svec_to_mat!(X,x,K)
        @test minimum(eigvals(X)) ≈ 1

        #X is positive definite.   eigenvalues should not change
        X = randpsd(rng,n)
        e = minimum(eigvals(X))
        Clarabel._mat_to_svec!(x,X,K)
        Clarabel.shift_to_cone!(K,x)
        Clarabel._svec_to_mat!(X,x,K)
        @test minimum(eigvals(X)) ≈ e

    end


    @testset "test_coneops_psdtrianglecone_update_scaling!" begin

        n = 5
        K = Clarabel.PSDTriangleCone(n)

        #X is negative definite.   Shift eigenvalues to 1
        S = randpsd(rng,n)
        Z = randpsd(rng,n)

        (s,z) = map(m->zeros(K.numel), 1:2)
        map((v,M)->Clarabel._mat_to_svec!(v,M,K),(s,z),(S,Z))

        Clarabel.update_scaling!(K,s,z)

        f = K.work
        R = f.R
        Rinv = f.Rinv
        Λ = Diagonal(f.λ)

        W = R*R'

        @test norm(W*Z*W' - S) ≈ 0  atol = sqrt(eps(FloatT))
        @test norm(R'*Z*R - Λ) ≈ 0  atol = sqrt(eps(FloatT))
        @test norm(Rinv*S*Rinv' - Λ) ≈ 0  atol = sqrt(eps(FloatT))

    end

    @testset "test_coneops_psdtrianglecone_step_length!" begin

        n = 10
        K = Clarabel.PSDTriangleCone(n)

        Z = randpsd(rng,n); dZ = randsym(rng,n)
        S = randpsd(rng,n); dS = randsym(rng,n)

        (s,z,ds,dz) = map(m->zeros(K.numel), 1:4)
        map((v,M)->Clarabel._mat_to_svec!(v,M,K),(s,z,ds,dz),(S,Z,dS,dZ))

        #compute internal scaling required for step calc
        Clarabel.update_scaling!(K,s,z)

        #Z direction only
        α = Clarabel.step_length(K,dz,ds.*0.,z,s)[1]
        @test minimum(eigvals(Z + α.*dZ)) ≈ 0.  atol = sqrt(eps(FloatT))

        #S direction only
        α = Clarabel.step_length(K,dz.*0,ds,z,s)[2]
        @test minimum(eigvals(S + α.*dS)) ≈ 0.  atol = sqrt(eps(FloatT))

        #joint
        (αz,αs) = Clarabel.step_length(K,dz,ds,z,s)
        eZ = eigvals(Z + αz.*dZ)
        eS = eigvals(S + αs.*dS)
        @test minimum([eZ;eS]) ≈ 0.  atol = sqrt(eps(FloatT))

        #unbounded
        dS .= randpsd(rng,n); dZ .= randpsd(rng,n)
        map((v,M)->Clarabel._mat_to_svec!(v,M,K),(ds,dz),(dS,dZ))
        (αz,αs) = Clarabel.step_length(K,dz,ds,z,s)
        @test min(αz,αs) ≈ floatmax(FloatT)  rtol = 10*eps(FloatT)

    end

    @testset "test NT scaling identities" begin

        n = 5
        K = Clarabel.PSDTriangleCone(n)

        (Z,S,V1,V2) = map(m->randpsd(rng,n), 1:4)
        (s,z,v1,v2) = map(m->zeros(K.numel), 1:4)
        map((v,M)->Clarabel._mat_to_svec!(v,M,K),(s,z,v1,v2),(S,Z,V1,V2))

        #compute internal scaling required for step calc
        Clarabel.update_scaling!(K,s,z)

        #check W^{-T}s = Wz = λ (λ is Diagonal)
        Clarabel.gemv_W!(K,:N,z,v1,one(FloatT),zero(FloatT)) #v1 = Wz
        Clarabel.gemv_Winv!(K,:T,s,v2,one(FloatT),zero(FloatT)) #v2 = W^{-T}s
        @test norm(v1-v2) ≈ 0   atol = 1e-10

        #check W^TW Z = S
        Clarabel.gemv_W!(K,:N,z,v1,one(FloatT),zero(FloatT)) #v1 = Wz
        Clarabel.gemv_W!(K,:T,v1,v2,one(FloatT),zero(FloatT)) #v2 = W^Tv1 = W^TWz
        @test norm(v2-s) ≈ 0   atol = 1e-10

        #check W^Tλ = s
        Λ = Matrix(Diagonal(K.work.λ))
        λ = Λ[triu(ones(n,n)) .== true]  #upper triangle (diagonal only)
        Clarabel.gemv_W!(K,:T,λ,v1,one(FloatT),zero(FloatT)) #v1 = W^Tλ
        @test norm(v1-s) ≈ 0   atol = 1e-10


    end

    @testset "test_coneops_psdtrianglecone_WtW_operations!" begin

        n = 5
        K = Clarabel.PSDTriangleCone(n)

        (Z,S,V1,V2,V3) = map(m->randpsd(rng,n), 1:5)
        (z,s,v1,v2,v3) = map(m->zeros(K.numel), 1:5)
        map((v,M)->Clarabel._mat_to_svec!(v,M,K),(s,z,v1,v2,v3),(S,Z,V1,V2,V3))

        #compute internal scaling required for step calc
        Clarabel.update_scaling!(K,s,z)

        R    = K.work.R
        Rinv = K.work.Rinv

        #compare different ways of multiplying v by W and W^T
        # v2 = W*v1
        Clarabel.gemv_W!(K,:N,v1,v2,one(FloatT),zero(FloatT))
        # v3 = W^T*v2
        Clarabel.gemv_W!(K,:T,v2,v3,one(FloatT),zero(FloatT))

        WtW = triu(ones(K.numel,K.numel))
        idxWtW = findall(WtW .!= 0)
        vecWtW = zeros(FloatT,length(idxWtW))
        Clarabel.get_WtW_block!(K,vecWtW)
        WtW[idxWtW] = vecWtW
        #make Symmetric for products
        WtWsym = Symmetric(WtW)

        @test norm(WtWsym*v1 - v3) ≈ 0   atol = 1e-8
        #now the inverse
        Clarabel.gemv_Winv!(K,:T,v1,v2,one(FloatT),zero(FloatT))
        # v3 = W^T*v2
        Clarabel.gemv_Winv!(K,:N,v2,v3,one(FloatT),zero(FloatT))
        @test norm(WtWsym\v1 - v3) ≈ 0   atol = 1e-8

    end



end
nothing
