using Convex, Clarabel, Test
@testset "Clarabel" begin
    Convex.ProblemDepot.run_tests([r"sdp_quantum_relative_entropy2_lowrank*"]; exclude=[r"mip"]) do p
        solve!(p, Convex.MOI.OptimizerWithAttributes(Clarabel.Optimizer, "verbose" => true))
    end
end
