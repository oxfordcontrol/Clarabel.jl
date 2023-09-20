using Test, LinearAlgebra, SparseArrays, Clarabel, Random
include("../testing_utils.jl")

rng = Random.MersenneTwister(242713)

FloatT = Float64

@testset "coneops_SecondOrderCone" begin

    @testset "test_coneops_soc_constructor" begin

        K = Clarabel.SecondOrderCone(5)
        @test Clarabel.numel(K)== 5
        @test Clarabel.degree(K) == 1
        @test_throws DomainError Clarabel.SecondOrderCone(-1)

    end

    @testset "test_coneops_scaled_unit_shift" begin

        n = 5 
        s = randn(n)
        s[1] = -1.
        K = Clarabel.SecondOrderCone(5)
        (m,_) = Clarabel.margins(K,s,Clarabel.PrimalCone)
        cor = m > 0. ? 0. : 1. -m
        Clarabel.scaled_unit_shift!(K,s,cor,Clarabel.PrimalCone)
        @test Clarabel._soc_residual(s) >= 0

    end

    @testset "test_coneops_W_construction" begin

        n = 5  #must be > 4 to avoid dense representation
        K = Clarabel.SecondOrderCone(n)
        s = randn(n)
        z = randn(n)
        (mz,_) = Clarabel.margins(K,z,Clarabel.DualCone)
        corz = mz > 0. ? 0. : 1. -mz
        (ms,_) = Clarabel.margins(K,s,Clarabel.PrimalCone)
        cors = ms > 0. ? 0. : 1. -ms
        Clarabel.scaled_unit_shift!(K,z,corz,Clarabel.DualCone)
        Clarabel.scaled_unit_shift!(K,s,cors,Clarabel.PrimalCone)
        μ = dot(s,z)
        scaling = Clarabel.PrimalDual

        # apply the scaling and then see if we get the
        # agreement abou the columns of W
        Clarabel.update_scaling!(K,s,z,μ,scaling)

        #extract values that should reconstruct W
        η = K.η
        d = K.sparse_data.d
        u = K.sparse_data.u
        v = K.sparse_data.v
        w = K.w
        D = I(n)*1.
        D[1,1] = d

        #this should be W^TW
        W2_A = η^2 .* (D + u*u' - v*v')

        #W^TW should agree with Hinv 
        J = -I(n).*1.; J[1,1] = 1.
        Hinv = η^2 .* (2*w*w' - J)

        @test norm(W2_A-Hinv) ≈ 0 atol = 1e-14

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
        @test norm(W_B-W_B') ≈ 0     atol = 1e-14

        #matrix and its inverse should agree 
        @test norm(W_B*W_Binv - I(n)) ≈ 0     atol = 1e-14

        # square should agree with the directly constructed one
        W2_B = W_B*W_B
        @test norm(W2_B-W2_A) ≈ 0     atol = 1e-12

        
    end


end
nothing
