using Test, LinearAlgebra, SparseArrays

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function basic_exp_data(Type::Type{T}) where {T <: AbstractFloat}

    #x is of dimension 7
    n = 7
    A1 = hcat(ones(T,1,3), zeros(T,1,4))        #ZeroCone
    b1 = 10.
    A2 = hcat(zeros(T,3,2), -Matrix(T(1.0)*I, 3, 3), zeros(T,3,2))       #NNCone
    b2 = zeros(T,3)
    A3 = zeros(T,3,7)               #expcone
    A3[1,1] = T(-1)
    A3[2,3] = T(-1)
    A3[3,5] = T(-1)
    b3 = zeros(3)

    c = T.([1.0; 0.5; -2.; -0.1; 1.0; 3.; 0.])
    P = spzeros(T,length(c), length(c))
    P = sparse(I(length(c)).*T(1e-1))

    A = sparse([A1;A2;A3])
    b = [b1;b2;b3]

    cones = [
        Clarabel.ZeroConeT(length(b1)),
        Clarabel.NonnegativeConeT(length(b2)),
        Clarabel.ExponentialConeT(),
    ]

    return (P,c,A,b,cones)
end


@testset "Basic ExponentialCone Tests" begin

    for FloatT in UnitTestFloats

        tol = FloatT(1e-3)

        @testset "Basic ExponentialCone Tests (T = $(FloatT))" begin

            @testset "feasible" begin

                P,c,A,b,cones = basic_exp_data(FloatT)
                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.info.status == Clarabel.SOLVED
                @test isapprox(
                norm(solver.solution.x -
                FloatT[ -9.425995201329599
                         4.828561507482018
                        14.59743362204262
                         1.0000012112102774
                         7.65314081561849
                         -29.99999978458479
                         -0.0]),
                zero(FloatT), atol=tol)
                @test isapprox(solver.info.cost_primal, FloatT(-54.41243965302268), atol=tol)

            end

        end      #end "Basic ExponentialCone Tests (FloatT)"

    end # UnitTestFloats

end #"Basic ExponentialCone Tests"

nothing
