using Test, LinearAlgebra, SparseArrays
using Clarabel, JuMP
FloatT = Float64
tol = FloatT(1e-3)

# A collection of tests from problems partly taken from the MOI Test
# collection, all of which caused issues with faulty infeasibility
# detection.

@testset "Infeasibility" begin

    @testset "Infeasibility checks (T = $(FloatT))" begin

        @testset "dual infeasible LP" begin
            #from MOI.Test.test_conic_SecondOrderCone_no_initial_bound

            model = JuMP.Model(Clarabel.Optimizer)
            set_optimizer_attribute(model, "verbose", true)

            @variable(model, t)
            @variable(model, x[1:2])
            @constraint(model, x .>= [3;4])
            @objective(model, Min, t)
            optimize!(model)
            @test JuMP.termination_status(model) == MOI.DUAL_INFEASIBLE

        end

        @testset "Const objective" begin
            #from MOI.Test.test_modification_delete_variable_with_single_variable_obj
            model = JuMP.Model(Clarabel.Optimizer)
            set_optimizer_attribute(model, "verbose", true)

            @variable(model, x)
            @variable(model, y)
            @constraint(model, x >= 1)
            @objective(model, Min, 1)
            optimize!(model)
            @test JuMP.termination_status(model) == MOI.OPTIMAL
            @test isapprox(JuMP.objective_value(model),1)

        end

        @testset "1 variable linear" begin
            #from MOI.Test.test_objective_ObjectiveFunction_VariableIndex
            model = JuMP.Model(Clarabel.Optimizer)
            set_optimizer_attribute(model, "verbose", true)

            @variable(model, x)
            @constraint(model, x >= 1)
            @objective(model, Min, x)
            optimize!(model)
            @test JuMP.termination_status(model) == MOI.OPTIMAL
            @test JuMP.objective_value(model) ≈ 1

        end

        @testset "linear + constant optimal" begin
            #from MOI.Test.test_objective_ObjectiveFunction_constant
            model = JuMP.Model(Clarabel.Optimizer)
            set_optimizer_attribute(model, "verbose", true)

            @variable(model, x)
            @constraint(model, x >= 1)
            @objective(model, Min, 2x + 1)
            optimize!(model)
            @test JuMP.termination_status(model) == MOI.OPTIMAL
            @test JuMP.objective_value(model) ≈ 3

        end


        @testset "repeated variable" begin
            #from MOI.Test.test_objective_ObjectiveFunction_duplicate_terms
            model = JuMP.Model(Clarabel.Optimizer)
            set_optimizer_attribute(model, "verbose", true)

            @variable(model, x)
            @constraint(model, c1, x >= 1)
            @objective(model, Min, 2x + x)
            optimize!(model)
            @test JuMP.termination_status(model) == MOI.OPTIMAL
            @test JuMP.objective_value(model) ≈ 3

        end

        #additional small tests for primal and dual infeasibility

        @testset "primal infeasible LP" begin

            model = JuMP.Model(Clarabel.Optimizer)
            set_optimizer_attribute(model, "verbose", true)

            @variable(model, x[1:2])
            @constraint(model, x[1] <= 1)
            @constraint(model, x[2] >= 3)
            @constraint(model, x[1] == x[2])
            @objective(model, Min, 1)
            optimize!(model)
            @test JuMP.termination_status(model) == MOI.INFEASIBLE

        end

        @testset "primal infeasible soc 1" begin

            model = JuMP.Model(Clarabel.Optimizer)
            set_optimizer_attribute(model, "verbose", true)

            @variable(model, t)
            @variable(model, x[1:2])
            @constraint(model, [t;x] in SecondOrderCone())
            @constraint(model, t <= -1)
            @objective(model, Min, t)
            optimize!(model)
            @test JuMP.termination_status(model) == MOI.INFEASIBLE

        end

        @testset "primal infeasible soc 2" begin

            model = JuMP.Model(Clarabel.Optimizer)
            set_optimizer_attribute(model, "verbose", true)

            @variable(model, t)
            @variable(model, x[1:2])
            @constraint(model, [t;x] in SecondOrderCone())
            @constraint(model, x[1] == 3)
            @constraint(model, t <= 1)
            @objective(model, Min, t)
            optimize!(model)
            @test JuMP.termination_status(model) == MOI.INFEASIBLE

        end

        @testset "dual infeasible soc 1" begin

            model = JuMP.Model(Clarabel.Optimizer)
            set_optimizer_attribute(model, "verbose", true)

            @variable(model, t[1:2])
            @variable(model, x[1:2])
            @constraint(model, [t[1];x] in SecondOrderCone())
            @constraint(model, x[1] == 3)
            @objective(model, Min, t[2])
            optimize!(model)
            @test JuMP.termination_status(model) == MOI.DUAL_INFEASIBLE

        end

        @testset "dual infeasible soc 2" begin

            model = JuMP.Model(Clarabel.Optimizer)
            set_optimizer_attribute(model, "verbose", true)

            @variable(model, t)
            @variable(model, x[1:3])
            @constraint(model, [t;x[1]+x[2];x[3]] in SecondOrderCone())
            @constraint(model, t <= 1)
            @objective(model, Min, x[1])
            optimize!(model)
            @test JuMP.termination_status(model) == MOI.DUAL_INFEASIBLE

        end


    end # UnitTestFloats
end
nothing
