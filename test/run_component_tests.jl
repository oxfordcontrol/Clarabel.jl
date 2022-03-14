using Random, Test, Pkg
rng = Random.MersenneTwister(12345)

@testset "Clarabel Unit Tests" begin

    #tests on small scale problems
    include("./UnitTests/test_coneops_psdcone.jl")


end
nothing
