using Random, Test

@testset "Clarabel Unit Tests" begin

    #tests on cones
    include("./UnitTests/test_coneops_psdtrianglecone.jl")
    include("./UnitTests/test_coneops_secondordercone.jl")

end
nothing
