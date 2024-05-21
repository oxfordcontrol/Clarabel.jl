using Random, Test
using Clarabel

UnitTestFloats = [Float64,BigFloat]

@testset "Clarabel Native Optimizer Testset" begin

    @testset "Basic Tests" begin

        #tests on small scale problems
        include("./OptTests/basic_unconstrained.jl")
        include("./OptTests/basic_eq_constrained.jl")
        include("./OptTests/basic_lp.jl")
        include("./OptTests/basic_qp.jl")
        include("./OptTests/basic_socp.jl")
        include("./OptTests/basic_exp.jl")
        include("./OptTests/basic_pow.jl")
        include("./OptTests/basic_sdp.jl")
        include("./OptTests/basic_genpow.jl")
        include("./OptTests/presolve.jl")
        include("./OptTests/data_updating.jl")
        include("./OptTests/sdp_chordal.jl")

    end

    @testset "Conic Optimization Tests" begin

        #tests on various cone problem types
        include("./OptTests/socp-lasso.jl")

    end

    @testset "Linear Solver Tests" begin

        #exercise different linear solverss
        include("./OptTests/linear_solvers.jl")

    end

end
nothing
