using Random, Test, Pkg
using Clarabel

rng = Random.MersenneTwister(12345)

@testset "Clarabel Native Optimizer Testset" begin

    @testset "Basic Tests" begin

        #tests on small scale problems
        include("./UnitTests/basic_lp.jl")
        include("./UnitTests/basic_qp.jl")
        include("./UnitTests/basic_socp.jl")
        include("./UnitTests/infeasibility.jl")

    end

    @testset "Conic Optimization Tests" begin

        #tests on various cone problem types
        include("./UnitTests/socp-lasso.jl")

    end

end
nothing
