using Test, LinearAlgebra, SparseArrays

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

@testset "Basic Unconstrained Tests" begin

    for FloatT in UnitTestFloats

        @testset "Unconstrained tests (T = $(FloatT))" begin

            tol = FloatT(1e-3)

            @testset "feasible" begin

                P = sparse(I(3).*one(FloatT))
                c = FloatT[1.,2.,-3.]
                A = sparse(zeros(FloatT,0,3)) #no constraints 
                b = FloatT[]
                cones = Clarabel.SupportedCone[]

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test isapprox(norm(solver.solution.x - (-c)), zero(FloatT), atol=tol)
                @test solver.solution.status == Clarabel.SOLVED

            end

            @testset "dual infeasible" begin

                P = sparse(I(3).*one(FloatT))
                P[1,1] = zero(FloatT)
                c = FloatT[1.,0.,0.]
                A = sparse(zeros(FloatT,0,3)) #no constraints 
                b = FloatT[]
                cones = Clarabel.SupportedCone[]

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.DUAL_INFEASIBLE

            end

        end      #end "tests (FloatT)"
    end
end # UnitTestFloats
