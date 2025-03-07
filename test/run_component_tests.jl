using Random, Test

@testset "Clarabel Unit Tests" begin

    #tests on cones
    include("./UnitTests/test_coneops_psdtrianglecone.jl")
    include("./UnitTests/test_coneops_secondordercone.jl")

    #tests on constructors 
    include("./UnitTests/test_constructors.jl")

    #tests on equilibration
    include("./UnitTests/test_equilibration_bounds.jl")

    #tests on json IO
    include("./UnitTests/test_json.jl")

    #tests on cone simplification and merging 
    include("./UnitTests/test_cones_new_collapsed.jl")

end
nothing
