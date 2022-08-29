using Convex, Clarabel, Test
@testset "Clarabel" begin
    Convex.ProblemDepot.run_tests([r""]; exclude=[r"mip"]) do p
        solve!(p, Convex.MOI.OptimizerWithAttributes(Clarabel.Optimizer, "verbose" => true))
    end
end
