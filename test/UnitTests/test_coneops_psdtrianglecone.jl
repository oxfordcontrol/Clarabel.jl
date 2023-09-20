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
        

    end

    @testset "test_coneops_psdtrianglecone_conversions" begin

        n = 5
        K = Clarabel.PSDTriangleCone(n)
        X = randsym(rng, n)
        Y = randsym(rng, n)
        Z = zeros(n,n)
        x = zeros(Clarabel.triangular_number(n))
        y = zeros(Clarabel.triangular_number(n))

        # check inner product identity
        map((v,M)->Clarabel._mat_to_svec!(v,M), (x,y), (X,Y))
        @test x'y - tr(X'Y)≈ 0  atol = 1e-12

        # check round trip
        Clarabel._mat_to_svec!(x,X)
        Clarabel._svec_to_mat!(Z,x)
        @test norm(X-Z) ≈ 0     atol = 1e-12


    end

    @testset "test_coneops_psdtrianglecone_circle_op" begin

        n = 5
        K = Clarabel.PSDTriangleCone(n)
        X1 = zeros(n,n)
        X2 = zeros(n,n)
        Y = randsym(rng, n)
        Z = randsym(rng, n)
        (x,y,z) = map(v->zeros(Clarabel.triangular_number(n)),1:3)
        map((v,M)->Clarabel._mat_to_svec!(v,M), (y,z), (Y,Z))
        K = Clarabel.PSDTriangleCone(n)

        X1 .= 0.5*(Y*Z + Z*Y)

        Clarabel.circ_op!(K,x,y,z)
        Clarabel._svec_to_mat!(X2,x)

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
        map((v,M)->Clarabel._mat_to_svec!(v,M),(x,z,λ,w),(X,Z,Λ,W))

        #Z = 1/2(ΛX + XΛ)
        Clarabel.circ_op!(K,z,λ,x)

        #W should now be the solution to 1/2(ΛW + WΛ) = Z
        K.data.λ .= λdiag        #diagonal internal scaling
        Clarabel.λ_inv_circ_op!(K,w,z)
        Clarabel._svec_to_mat!(W,w)

        #now we should have x = w
        @test norm(x  - w) ≈ 0 atol = 100*eps(FloatT)
        #now we should have x = w
        @test norm(X  - W) ≈ 0 atol = 100*eps(FloatT)

    end



    @testset "test_coneops_psdtrianglecone_scaled_unit_shift!" begin

        n = 5
        a = 0.12345
        K = Clarabel.PSDTriangleCone(n)
        X = randsym(rng,n)
        X .= 1. * 0.
        x = zeros(K.numel)

        XplusaI = X + a*I(n)
        Clarabel._mat_to_svec!(x,X)
        Clarabel.scaled_unit_shift!(K,x,a,Clarabel.PrimalCone)
        Clarabel._svec_to_mat!(X,x)

        @test norm(X - XplusaI) ≈ 0

    end


    @testset "test_coneops_psdtrianglecone_update_scaling!" begin

        n = 5
        K = Clarabel.PSDTriangleCone(n)

        #X is negative definite.   Shift eigenvalues to 1
        S = randpsd(rng,n)
        Z = randpsd(rng,n)

        (s,z) = map(m->zeros(K.numel), 1:2)
        map((v,M)->Clarabel._mat_to_svec!(v,M),(s,z),(S,Z))

        μ = 0.0 #placeholder value, not used
        strategy = Clarabel.PrimalDual
        Clarabel.update_scaling!(K,s,z,μ,strategy)

        f = K.data
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
        settings = Clarabel.Settings{Float64}()

        Z = randpsd(rng,n); dZ = randsym(rng,n)
        S = randpsd(rng,n); dS = randsym(rng,n)

        (s,z,ds,dz) = map(m->zeros(K.numel), 1:4)
        map((v,M)->Clarabel._mat_to_svec!(v,M),(s,z,ds,dz),(S,Z,dS,dZ))

        #compute internal scaling required for step calc
        μ = 0.0 #placeholder value, not used
        strategy = Clarabel.PrimalDual
        Clarabel.update_scaling!(K,s,z,μ,strategy)

        #Z direction only
        α = Clarabel.step_length(K,dz,ds.*0.,z,s,settings,1.0)[1]
        @test minimum(eigvals(Z + α.*dZ)) ≈ 0.  atol = sqrt(eps(FloatT))

        #S direction only
        α = Clarabel.step_length(K,dz.*0,ds,z,s,settings,1.0)[2]
        @test minimum(eigvals(S + α.*dS)) ≈ 0.  atol = sqrt(eps(FloatT))

        #joint
        (αz,αs) = Clarabel.step_length(K,dz,ds,z,s,settings,1.0)
        eZ = eigvals(Z + αz.*dZ)
        eS = eigvals(S + αs.*dS)
        @test minimum([eZ;eS]) ≈ 0.  atol = sqrt(eps(FloatT))

        #should reach maximum step 
        dS .= randpsd(rng,n); dZ .= randpsd(rng,n)
        map((v,M)->Clarabel._mat_to_svec!(v,M),(ds,dz),(dS,dZ))
        (αz,αs) = Clarabel.step_length(K,dz,ds,z,s,settings,1.0)
        @test min(αz,αs) ≈ 1.0  rtol = 10*eps(FloatT)

    end

    @testset "test NT scaling identities" begin

        n = 5
        K = Clarabel.PSDTriangleCone(n)

        (Z,S,V1,V2) = map(m->randpsd(rng,n), 1:4)
        (s,z,v1,v2) = map(m->zeros(K.numel), 1:4)
        map((v,M)->Clarabel._mat_to_svec!(v,M),(s,z,v1,v2),(S,Z,V1,V2))

        #compute internal scaling required for step calc
        μ = 0.0 #placeholder value, not used
        strategy = Clarabel.PrimalDual
        Clarabel.update_scaling!(K,s,z,μ,strategy)

        #check W^{-T}s = Wz = λ (λ is Diagonal)
        Clarabel.mul_W!(K,:N,v1,z,one(FloatT),zero(FloatT)) #v1 = Wz
        Clarabel.mul_Winv!(K,:T,v2,s,one(FloatT),zero(FloatT)) #v2 = W^{-T}s
        @test norm(v1-v2) ≈ 0   atol = 1e-10

        #check W^TW Z = S
        Clarabel.mul_W!(K,:N,v1,z,one(FloatT),zero(FloatT)) #v1 = Wz
        Clarabel.mul_W!(K,:T,v2,v1,one(FloatT),zero(FloatT)) #v2 = W^Tv1 = W^TWz
        @test norm(v2-s) ≈ 0   atol = 1e-10

        #check W^Tλ = s
        Λ = Matrix(Diagonal(K.data.λ))
        λ = Λ[triu(ones(n,n)) .== true]  #upper triangle (diagonal only)
        Clarabel.mul_W!(K,:T,v1,λ,one(FloatT),zero(FloatT)) #v1 = W^Tλ
        @test norm(v1-s) ≈ 0   atol = 1e-10


    end

    @testset "test_coneops_psdtrianglecone_Hs_operations!" begin

        n = 5
        K = Clarabel.PSDTriangleCone(n)

        (Z,S,V1,V2,V3) = map(m->randpsd(rng,n), 1:5)
        (z,s,v1,v2,v3) = map(m->zeros(K.numel), 1:5)
        map((v,M)->Clarabel._mat_to_svec!(v,M),(s,z,v1,v2,v3),(S,Z,V1,V2,V3))

        #compute internal scaling required for step calc
        μ = 0.0 #placeholder value, not used
        strategy = Clarabel.PrimalDual
        Clarabel.update_scaling!(K,s,z,μ,strategy)

        R    = K.data.R
        Rinv = K.data.Rinv

        #compare different ways of multiplying v by W and W^T
        # v2 = W*v1
        Clarabel.mul_W!(K,:N,v2,v1,one(FloatT),zero(FloatT))
        # v3 = W^T*v2
        Clarabel.mul_W!(K,:T,v3,v2,one(FloatT),zero(FloatT))

        Hs = triu(ones(K.numel,K.numel))
        idxHs = findall(Hs .!= 0)
        vecHs = zeros(FloatT,length(idxHs))
        Clarabel.get_Hs!(K,vecHs)
        Hs[idxHs] = vecHs
        #make Symmetric for products
        Hssym = Symmetric(Hs)

        @test norm(Hssym*v1 - v3) ≈ 0   atol = 1e-8
        #now the inverse
        Clarabel.mul_Winv!(K,:T,v2,v1,one(FloatT),zero(FloatT))
        # v3 = W^T*v2
        Clarabel.mul_Winv!(K,:N,v3,v2,one(FloatT),zero(FloatT))
        @test norm(Hssym\v1 - v3) ≈ 0   atol = 1e-8

    end



end
nothing
