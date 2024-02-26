using Test, LinearAlgebra, SparseArrays

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function basic_LP_data(Type::Type{T}) where {T <: AbstractFloat}

    P = spzeros(T,3,3)
    A = sparse(I(3)*T(1.))
    A = [A;-A].*2
    c = T[3.;-2.;1.]
    b = ones(T,6)
    cones = [Clarabel.NonnegativeConeT(3), Clarabel.NonnegativeConeT(3)]

    return (P,c,A,b,cones)
end


@testset "Basic LP Tests" begin

    for FloatT in UnitTestFloats

        @testset "Basic LP Tests (T = $(FloatT))" begin

            tol = FloatT(1e-3)
            @testset "feasible" begin

                P,c,A,b,cones = basic_LP_data(FloatT)
                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.SOLVED
                @test isapprox(norm(solver.solution.x - FloatT[-0.5; 0.5; -0.5]), zero(FloatT), atol=tol)
                @test isapprox(solver.solution.obj_val, FloatT(-3.), atol=tol)
                @test isapprox(solver.solution.obj_val_dual, FloatT(-3.), atol=tol)


            end

            @testset "primal infeasible" begin

                P,c,A,b,cones = basic_LP_data(FloatT)
                b[1] = -1
                b[4] = -1

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.PRIMAL_INFEASIBLE
                @test isnan(solver.solution.obj_val)
                @test isnan(solver.solution.obj_val_dual)

            end

            @testset "dual infeasible" begin

                P,c,A,b,cones = basic_LP_data(FloatT)
                A[4,1] = 1.  #swap lower bound on first variable to redundant upper bound
                c .= FloatT[1.;0;0]

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.DUAL_INFEASIBLE
                @test isnan(solver.solution.obj_val)
                @test isnan(solver.solution.obj_val_dual)

            end

            @testset "dual infeasible (ill conditioned)" begin

                P,c,A,b,cones = basic_LP_data(FloatT)
                A[1,1] = eps(FloatT)
                A[4,1] = 0.0
                c .= FloatT[1.;0;0]

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.DUAL_INFEASIBLE
                @test isnan(solver.solution.obj_val)
                @test isnan(solver.solution.obj_val_dual)

            end

        end      #end "Basic LP Tests (FloatT)"
    end
end # UnitTestFloats
