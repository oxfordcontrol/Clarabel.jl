using Test, LinearAlgebra, SparseArrays

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function equilibration_test_data(Type::Type{T}) where {T <: AbstractFloat}
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


@testset "Equilibration Tests" begin

    for FloatT in UnitTestFloats

        @testset "Equilibration Test (T = $(FloatT))" begin

            @testset "equilibrate lower bound" begin

                P,c,A,b,cones = equilibration_test_data(FloatT)
                settings = Clarabel.Settings{FloatT}()

                P[1,1] = FloatT(1e-15)

                solver   = Clarabel.Solver(P,c,A,b,cones,settings)

                d = solver.data.equilibration.d
                e = solver.data.equilibration.e

                @test minimum(d) >= settings.equilibrate_min_scaling
                @test minimum(e) >= settings.equilibrate_min_scaling
                @test maximum(d) <= settings.equilibrate_max_scaling
                @test maximum(e) <= settings.equilibrate_max_scaling
            
            end

            @testset "equilibrate upper bound" begin

                P,c,A,b,cones = equilibration_test_data(FloatT)
                settings = Clarabel.Settings{FloatT}()

                A[1,1] = FloatT(1e+15)

                solver   = Clarabel.Solver(P,c,A,b,cones,settings)

                d = solver.data.equilibration.d
                e = solver.data.equilibration.e

                @test minimum(d) >= settings.equilibrate_min_scaling
                @test minimum(e) >= settings.equilibrate_min_scaling
                @test maximum(d) <= settings.equilibrate_max_scaling
                @test maximum(e) <= settings.equilibrate_max_scaling
            
            
            end

            @testset "equilibrate zero rows" begin

                P,c,A,b,cones = equilibration_test_data(FloatT)
                A.nzval .= zero(FloatT)
                settings = Clarabel.Settings{FloatT}()

                solver   = Clarabel.Solver(P,c,A,b,cones,settings)

                d = solver.data.equilibration.d
                e = solver.data.equilibration.e

                @test all(e .== one(FloatT)) == true
            
            end

        end      #end "Equilibration Test (FloatT)"

    end # UnitTestFloats

end #"Equilibration Tests"

nothing
