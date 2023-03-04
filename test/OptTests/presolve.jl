using Test, LinearAlgebra, SparseArrays

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function presolver_test_data(Type::Type{T}) where {T <: AbstractFloat}

    P = sparse(I(3)*one(T))
    A = sparse(I(3)*one(T))
    A = [A;-A].*2
    c = T[3.;-2.;1.]
    b = ones(T,6)
    cones = [Clarabel.NonnegativeConeT(3), Clarabel.NonnegativeConeT(3)]

    return (P,c,A,b,cones)
end


@testset "Presolver Tests" begin

    for FloatT in UnitTestFloats

        #force default infinite bounds in solver 
        Clarabel.default_infinity()

        @testset "Presolver tests (T = $(FloatT))" begin
                
            @testset "single unbounded constraint" begin

                # Just one bound to zero 
                P,c,A,b,cones = presolver_test_data(FloatT)
                solver   = Clarabel.Solver{FloatT}()
                settings = Clarabel.Settings{FloatT}()
                b[4] = 1e30
                Clarabel.setup!(solver,P,c,A,b,cones)
                Clarabel.solve!(solver)
                @test solver.solution.status == Clarabel.SOLVED
                @test (length(solver.variables.z) == 5)
                @test (all(solver.solution.z[4] .== zero(T)))
                @test (all(solver.solution.s[4] .== Clarabel.get_infinity()))
           
            end

            @testset "completely redundant cone" begin

                # all bounds on first cone huge
                P,c,A,b,cones = presolver_test_data(FloatT)
                solver   = Clarabel.Solver{FloatT}()
                settings = Clarabel.Settings{FloatT}()
                b[1:3] .= 1e30
                Clarabel.setup!(solver,P,c,A,b,cones)
                Clarabel.solve!(solver)
                @test solver.solution.status == Clarabel.SOLVED
                @test (length(solver.variables.z) == 3)
                @test (all(solver.solution.z[1:3] .== zero(T)))
                @test (all(solver.solution.s[1:3] .== Clarabel.get_infinity()))
                @test isapprox(norm(solver.solution.x .- FloatT[-0.5; 2; -0.5]), zero(FloatT), atol=tol)

            end

            @testset "every constraint redundant" begin

                # all bounds on first cone huge
                P,c,A,b,cones = presolver_test_data(FloatT)
                solver   = Clarabel.Solver{FloatT}()
                settings = Clarabel.Settings{FloatT}()
                b[1:6] .= 1e30
                Clarabel.setup!(solver,P,c,A,b,cones)
                Clarabel.solve!(solver)
                @test solver.solution.status == Clarabel.SOLVED
                @test (length(solver.variables.z) == 0)
                @test isapprox(norm(solver.solution.x .+ c), zero(FloatT), atol=tol)

            end

            @testset "settable bound" begin

                thebound = Clarabel.get_infinity()
                Clarabel.set_infinity(1e21)
                @test (Clarabel.get_infinity() == 1e21)
                Clarabel.set_infinity(thebound) 
                @test (Clarabel.get_infinity() == thebound)  

            end

        end      #end Presolver Tests (FloatT)"
    end # UnitTestFloats
end