using Test
using Clarabel

@testset "Clarabel Interfaces Testset" begin

    #MOI test suite
    include("./Interfaces/MOI_wrapper_tests.jl")

    @testset "Clarabel-JuMP" begin
        include("./Interfaces/JuMP_infeasibility_tests.jl")
    end

end
nothing
