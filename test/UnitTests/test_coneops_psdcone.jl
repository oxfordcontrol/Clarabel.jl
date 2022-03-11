using Test, LinearAlgebra, SparseArrays, Clarabel

@testset "coneops_psdcone" begin


    function randsym(n)
        A = randn(n,n   )
        A = A+A'
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

        Clarabel.circle_op!(K,x,y,z)
        X .= 0.5*(Y*Z + Z*Y)

        @test tr(X) ≈ dot(y,z)
        @test norm(x - X[:]) ≈ 0

    end

    @testset "test_coneops_psdcone_inv_circle_op_small" begin

        n = 2
        X = [-2. 3; 3. -5.]
        Y = [1. 2; 2. 1]
        Z = [-2. -4; -4. 6.]
        (x,y,z,w) = map(m->reshape(m,:), (X,Y,Z,W))
        K = Clarabel.PSDCone(n^2)
        W = zeros(n,n)
        w = reshape(W,:)

        #Z = 1/2(YX + XY)
        Clarabel.circle_op!(K,z,x,y)

        #W should now be the solution to 1/2(YW + WY) = Z
        Clarabel.inv_circle_op!(K,w,y,z)

        #now we should have x = w

        @test norm(R[:]  - Z[:]) ≈ 0

    end

    @testset "test_coneops_psdcone_inv_circle_op" begin

        n = 2
        X = zeros(n,n)
        Y = [1. 2; 2. 1]
        Z = [-2. 3; 3. -3.]
        (x,y,z) = map(m->reshape(m,:), (X,Y,Z))
        K = Clarabel.PSDCone(n^2)
        W = zeros(n,n)
        w = reshape(W,:)


        Clarabel.inv_circle_op!(K,w,y,x)

        #X should now be the solution to 1/2(YX + XY) = Z
        R = (1/2)*(Y*X + X*Y)

        @test norm(R[:]  - Z[:]) ≈ 0

    end

    # @testset "test_coneops_psdcone_inv_circle_op" begin
    #
    #     n = 5
    #     X = zeros(n,n)
    #     Y = randn(n,n)
    #     Z = randn(n,n)
    #     x = X[:]; y = Y[:]; z = Z[:]
    #     K = Clarabel.PSDCone(n^2)
    #
    #     Clarabel.inv_circle_op!(K,x,y,z)
    #
    #     #we should have x = y\z.   Then y ∘ (y\z) = y ∘ x should be z again
    #     Ztest = zeros(n,n)
    #     ztest = reshape(Ztest,:,1)
    #     Clarabel.circle_op!(K,Ztest[:],y,x)
    #
    #     @test norm(Z[:] - Ztest[:]) ≈ 0
    #
    # end

end
nothing
