using Test, LinearAlgebra, SparseArrays, Clarabel, Random
include("../testing_utils.jl")

rng = Random.MersenneTwister(242713)

FloatT = Float64

@testset "coneops_SecondOrderCone" begin

    @testset "test_coneops_soc_constructor" begin

        K = Clarabel.SecondOrderCone(5)
        @test Clarabel.numel(K)== 5
        @test Clarabel.degree(K) == 1
        @test_throws DomainError Clarabel.PSDTriangleCone(-1)
        @test_throws DomainError Clarabel.PSDTriangleCone(0)

    end

    @testset "test_coneops_bring_to_cone" begin

        n = 5 
        s = randn(n)
        s[1] = -1.
        K = Clarabel.SecondOrderCone(5)
        Clarabel.shift_to_cone!(K,s)
        @test Clarabel._soc_residual(s) >= 0

    end

    @testset "test_coneops_W_construction" begin

        n = 5
        K = Clarabel.SecondOrderCone(n)
        s = randn(n)
        z = randn(n)
        Clarabel.shift_to_cone!(K,s)
        Clarabel.shift_to_cone!(K,z)
        μ = dot(s,z)
        scaling = Clarabel.PrimalDual

        # apply the scaling and then see if we get the
        # agreement abou the columns of W
        Clarabel.update_scaling!(K,s,z,μ,scaling)

        #extract values that should reconstruct W
        η = K.η; d = K.d
        u = K.u; v = K.v
        D = I(n)*1.
        D[1,1] = d

        #this should be W^TW
        W2_A = η^2 .* (D + u*u' - v*v')

        #now get W and Winv by repeated multiplication
        W_Binv = zeros(n,n)
        W_B = zeros(n,n)
        for i = 1:n
            e = zeros(n); e[i] = 1;
            out = zeros(n)
            Clarabel.mul_W!(K,:N,out,e,1.,0.)
            W_B[:,i] .= out
            Clarabel.mul_Winv!(K,:N,out,e,1.,0.)
            W_Binv[:,i] .= out
        end

        # W_B should be symmetric
        @test norm(W_B-W_B') ≈ 0     atol = 1e-15

        #matrix and it's inverse should agree 
        @test norm(W_B*W_Binv - I(n)) ≈ 0     atol = 1e-15

        # square should agree with the directly constructed on
        W2_B = W_B*W_B
        @test norm(W2_B-W2_A) ≈ 0     atol = 1e-15

        
    end


end
nothing
