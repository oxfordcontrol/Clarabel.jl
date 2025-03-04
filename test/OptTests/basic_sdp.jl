using Test, LinearAlgebra, SparseArrays

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function basic_SDP_data(Type::Type{T}) where {T <: AbstractFloat}

    # problem will be 3x3, so upper triangle 
    # of problem data has 6 entries 

    P = sparse(I(6).*one(T))
    c = zeros(T,6)
    A = sparse(I(6).*one(T))
    b = T[-3., 1., 4., 1., 2., 5.]   #triu of some indefinite matrix
    
    cones = Clarabel.SupportedCone[Clarabel.PSDTriangleConeT(3)]

    return (P,c,A,b,cones)

end

@testset "Basic SDP Tests" begin

    for FloatT in UnitTestFloats

        tol = FloatT(1e-3)

        @testset "Basic SDP Tests (T = $(FloatT))" begin

            @testset "feasible" begin

                P,c,A,b,cones = basic_SDP_data(FloatT)

                solver   = Clarabel.Solver(P,c,A,b,cones)
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

            @testset "empty SDP cone" begin

                P,c,A,b,cones = basic_SDP_data(FloatT)
                push!(cones,Clarabel.PSDTriangleConeT(0))

                solver   = Clarabel.Solver(P,c,A,b,cones)
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

            @testset "primal infeasible" begin

                P,c,A,b,cones = basic_SDP_data(FloatT)
               
                #this adds a negative definiteness constraint to x
                A = [A;-A]
                b = [b;zeros(length(b))]
                cones = repeat(cones,2)

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.PRIMAL_INFEASIBLE

            end

            @testset "1x1 sdp autoconversion" begin 
                using Clarabel, LinearAlgebra, SparseArrays

                P = sparse(I(1).*one(FloatT))
                q = zeros(FloatT,1)

                A = sparse(I(1).*one(FloatT))
                b = ones(FloatT,1)

                cones = [Clarabel.PSDTriangleConeT(1)]

                solver   = Clarabel.Solver(P,q,A,b,cones)
                Clarabel.solve!(solver)
                @test isapprox(norm(solver.solution.x - FloatT[0.]), zero(FloatT), atol=tol)
                @test isapprox(solver.solution.obj_val, FloatT(0.), atol=tol)
                @test isapprox(solver.solution.obj_val_dual, FloatT(0.), atol=tol)


            end

        end      #end "Basic SDP Tests (FloatT)"

    end # UnitTestFloats

end #"Basic SDP Tests"

nothing

