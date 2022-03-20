using Random, Test, Pkg

@testset "Clarabel Unit Tests" begin

    #tests on small scale problems
    include("./UnitTests/test_coneops_psdtrianglecone.jl")

end
nothing
