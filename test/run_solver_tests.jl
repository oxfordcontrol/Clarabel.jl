using Random, Test, Pkg
using Clarabel

@testset "Clarabel Native Optimizer Testset" begin

    @testset "Basic Tests" begin

        #tests on small scale problems
        include("./OptTests/basic_lp.jl")
        include("./OptTests/basic_qp.jl")
        include("./OptTests/basic_socp.jl")
        include("./OptTests/infeasibility.jl")

    end

    @testset "Conic Optimization Tests" begin

        #tests on various cone problem types
        include("./OptTests/socp-lasso.jl")

    end

end
nothing
