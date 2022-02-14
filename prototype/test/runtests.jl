using Random, Test
include("./UnitTests/test_utils.jl")
include("../IPSolver.jl")

# Define the types to run the unit tests with
UnitTestFloats = [Float32; Float64; BigFloat]
UnitTestFloats = [Float64]

@testset "All Clarabel Tests" begin

    @testset "Clarabel Tests" begin

        # Tests running calls to the optimizer
        # for various conic problem types
        include("./run_solver_tests.jl")

        # Tests running component level tests
        include("./run_component_tests.jl")

    end

end
nothing
