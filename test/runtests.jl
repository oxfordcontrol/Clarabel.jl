 using Test, Clarabel
include("./testing_utils.jl")

@testset "Clarabel Tests" begin

    # Tests running calls to the optimizer
    # for various conic problem types
    include("./run_solver_tests.jl")

    # Tests running component level tests
    include("./run_component_tests.jl")

    #MathOptInterface / JuMP etc
    include("./run_interface_tests.jl")

end
