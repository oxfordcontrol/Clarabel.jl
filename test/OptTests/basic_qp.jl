using Test, LinearAlgebra, SparseArrays

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function basic_QP_data(Type::Type{T}) where {T <: AbstractFloat}
    P = sparse(T[4. 1;1 2])
    c = T[1; 1.]
    A = sparse(T[1. 1;1 0; 0 1])
    l = T[1.;0;0]
    u = T[1.;0.7;0.7]

    cones = [Clarabel.NonnegativeConeT(3), Clarabel.NonnegativeConeT(3)]

    A = [-A;A]
    b = [-l;u]

    return (P,c,A,b,cones)
end


function basic_QP_data_dualinf(Type::Type{T}) where {T <: AbstractFloat}

    #x = [1;-1] is in ker(P) and always feasible
    P = sparse(T[1. 1.;1. 1.])
    c = T[1; -1.]
    A = sparse(T[1. 1;1. 0;])
    b = T[1.;1]
    cones = [Clarabel.NonnegativeConeT(2)]

    return (P,c,A,b,cones)
end


@testset "Basic QP Tests" begin

    for FloatT in UnitTestFloats

        tol = FloatT(1e-3)

        @testset "Basic QP Tests (T = $(FloatT))" begin


            @testset "univariate" begin 
                using Clarabel, LinearAlgebra, SparseArrays

                P = sparse(I(1).*one(FloatT))
                q = zeros(FloatT,1)

                A = sparse(I(1).*one(FloatT))
                b = ones(FloatT,1)

                cones = [Clarabel.NonnegativeConeT(1)]

                solver   = Clarabel.Solver(P,q,A,b,cones)
                Clarabel.solve!(solver)
                @test isapprox(norm(solver.solution.x - FloatT[0.]), zero(FloatT), atol=tol)
                @test isapprox(solver.solution.obj_val, FloatT(0.), atol=tol)
                @test isapprox(solver.solution.obj_val_dual, FloatT(0.), atol=tol)


            end

            @testset "feasible" begin

                P,c,A,b,cones = basic_QP_data(FloatT)
                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.SOLVED
                @test isapprox(norm(solver.solution.x - FloatT[0.3; 0.7]), zero(FloatT), atol=tol)
                @test isapprox(solver.solution.obj_val, FloatT(1.8800000298331538), atol=tol)
                @test isapprox(solver.solution.obj_val_dual, FloatT(1.8800000298331538), atol=tol)

            end

            @testset "primal infeasible" begin

                P,c,A,b,cones = basic_QP_data(FloatT)
                b[1] = -one(FloatT)
                b[4] = -one(FloatT)

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.PRIMAL_INFEASIBLE
                @test isnan(solver.solution.obj_val)
                @test isnan(solver.solution.obj_val_dual)

            end

            @testset "dual infeasible" begin

                P,c,A,b,cones = basic_QP_data_dualinf(FloatT)
                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.DUAL_INFEASIBLE
                @test isnan(solver.solution.obj_val)
                @test isnan(solver.solution.obj_val_dual)
            end

            @testset "dual infeasible (non-QSD KKT)" begin

                P,c,A,b,cones = basic_QP_data_dualinf(FloatT)
                A = A[1:1,:]
                b = b[1:1]
                cones = [Clarabel.NonnegativeConeT(1)]

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.DUAL_INFEASIBLE
                @test isnan(solver.solution.obj_val)
                @test isnan(solver.solution.obj_val_dual)
            end

        end      #end "Basic QP Tests (FloatT)"

    end # UnitTestFloats

end #"Basic QP Tests"

nothing
