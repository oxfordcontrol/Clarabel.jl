using Test, LinearAlgebra, SparseArrays

#Test each of these solvers, but only for Float64

#NB: mkl fails here because of some weird library
#issue caused by loading conflicting BLAS libraries.
#:mkl works, but not if you are also testing
#JuMP.  Issue here:
# https://github.com/JuliaSparse/Pardiso.jl/issues/88

SolverTypes = [:qdldl, :cholmod]

for SolverType in SolverTypes

    @testset "Linear solve using $(SolverType)" begin

        @testset "QP" begin

            settings = Clarabel.Settings(direct_solve_method = SolverType)
            P, c, A, b, cones = basic_QP_data(Float64)
            solver = Clarabel.Solver(P, c, A, b, cones, settings)
            Clarabel.solve!(solver)

            @test solver.solution.status == Clarabel.SOLVED
            @test isapprox(norm(solver.solution.x - Float64[0.3; 0.7]), zero(Float64), atol=tol)
            @test isapprox(solver.solution.obj_val, Float64(1.8800000298331538), atol=tol)

        end

        @testset "SOCP" begin

            settings = Clarabel.Settings(direct_solve_method = SolverType)
            P, c, A, b, cones = basic_SOCP_data(Float64)
            solver = Clarabel.Solver(P, c, A, b, cones, settings)
            Clarabel.solve!(solver)

            @test solver.solution.status == Clarabel.SOLVED
            @test isapprox(
                norm(solver.solution.x -
                FloatT[-0.127715, 0.108242, -0.067784]),
                zero(FloatT), atol=tol
            )
            @test isapprox(solver.solution.obj_val, FloatT(-0.148520), atol=tol)

        end

        @testset "SDP" begin

            settings = Clarabel.Settings(direct_solve_method = SolverType)
            P, c, A, b, cones = basic_SDP_data(Float64)
            solver = Clarabel.Solver(P, c, A, b, cones, settings)
            Clarabel.solve!(solver)

            refsol =  FloatT[
                -3.0729833267361095
                 0.3696004167288786
                -0.022226685581313674
                 0.31441213129613066
                -0.026739700851545107
                -0.016084530571308823
            ]

            @test solver.solution.status == Clarabel.SOLVED
            @test isapprox(norm(solver.solution.x - refsol), zero(FloatT), atol=tol)
            @test isapprox(solver.solution.obj_val, FloatT(4.840076866013861), atol=tol)

        end

    end      #end Linear solver tests

end # SolverTypes

nothing
