using Test, LinearAlgebra, SparseArrays

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])

function updating_test_data(Type::Type{T}) where {T <: AbstractFloat}
    P = sparse(T[4. 1;1 2])
    q = T[1; 1.]
    A = sparse(T[1 0; 0 1])
    l = T[-1;-1]
    u = T[1;1]

    cones = [Clarabel.NonnegativeConeT(2), Clarabel.NonnegativeConeT(2)]

    A = [-A;A]
    b = [-l;u]

    settings = Clarabel.Settings{T}()
    settings.presolve_enable = false
    settings.chordal_decomposition_enable = false

    return (P,q,A,b,cones,settings)
end


@testset "Data Updating Tests" begin

    for FloatT in UnitTestFloats

        tol = FloatT(1e-7)

        @testset "Data Updating Tests (T = $(FloatT))" begin

            @testset "updata P (matrix form)" begin 
                
                #original problem 
                P,q,A,b,cones,settings = updating_test_data(FloatT)
                solver1   = Clarabel.Solver(P,q,A,b,cones,settings)
                Clarabel.solve!(solver1)

                #change P and re-solve
                P2 = deepcopy(P)
                P2[1,1] = 100

                #revised original solver 
                Clarabel.update_P!(solver1,triu(P2))
                Clarabel.solve!(solver1)

                #new solver 
                solver2 = Clarabel.Solver(P2,q,A,b,cones,settings)
                Clarabel.solve!(solver2)

                @test isapprox(norm(solver1.solution.x - solver2.solution.x), zero(FloatT), atol=tol)

            end

            @testset "updata P (vector form)" begin 
                
                #original problem 
                P,q,A,b,cones,settings = updating_test_data(FloatT)
                solver1   = Clarabel.Solver(P,q,A,b,cones,settings)
                Clarabel.solve!(solver1)

                #change P and re-solve
                P2 = deepcopy(P)
                P2[1,1] = 100

                #revised original solver 
                Clarabel.update_P!(solver1,triu(P2).nzval)
                Clarabel.solve!(solver1)

                #new solver 
                solver2 = Clarabel.Solver(P2,q,A,b,cones,settings)
                Clarabel.solve!(solver2)

                @test isapprox(norm(solver1.solution.x - solver2.solution.x), zero(FloatT), atol=tol)

            end

            @testset "updata P (tuple)" begin 
                
                #original problem 
                P,q,A,b,cones,settings = updating_test_data(FloatT)
                solver1   = Clarabel.Solver(P,q,A,b,cones,settings)
                Clarabel.solve!(solver1)

                #revised original solver
                values = FloatT[3.,5.]
                index  = [2,3]
                Pdata  = zip(index,values)
                Clarabel.update_P!(solver1,Pdata)
                Clarabel.solve!(solver1)

                #new solver 
                P2 = sparse(FloatT[4. 3; 0 5])
                solver2 = Clarabel.Solver(P2,q,A,b,cones,settings)
                Clarabel.solve!(solver2)

                @test isapprox(norm(solver1.solution.x - solver2.solution.x), zero(FloatT), atol=tol)

            end

            @testset "updata A (matrix form)" begin 
                
                #original problem 
                P,q,A,b,cones,settings = updating_test_data(FloatT)
                solver1   = Clarabel.Solver(P,q,A,b,cones,settings)
                Clarabel.solve!(solver1)

                #change A and re-solve 
                A2 = deepcopy(A)
                A2[2,2] = -1000.

                #revised original solver 
                Clarabel.update_A!(solver1,A2)
                Clarabel.solve!(solver1)

                #new solver 
                solver2 = Clarabel.Solver(P,q,A2,b,cones,settings)
                Clarabel.solve!(solver2)

                @test isapprox(norm(solver1.solution.x - solver2.solution.x), zero(FloatT), atol=tol)

            end

            @testset "updata A (vector form)" begin 
                
                #original problem 
                P,q,A,b,cones,settings = updating_test_data(FloatT)
                solver1   = Clarabel.Solver(P,q,A,b,cones,settings)
                Clarabel.solve!(solver1)

                #change A and re-solve 
                A2 = deepcopy(A)
                A2[2,2] = -1000.

                #revised original solver 
                Clarabel.update_A!(solver1,A2.nzval)
                Clarabel.solve!(solver1)

                #new solver 
                solver2 = Clarabel.Solver(P,q,A2,b,cones,settings)
                Clarabel.solve!(solver2)

                @test isapprox(norm(solver1.solution.x - solver2.solution.x), zero(FloatT), atol=tol)

            end

            @testset "updata A (tuple)" begin 
                
                #original problem 
                P,q,A,b,cones,settings = updating_test_data(FloatT)
                solver1   = Clarabel.Solver(P,q,A,b,cones,settings)
                Clarabel.solve!(solver1)

                #revised original solver
                values = FloatT[0.5,-0.5]
                index  = [2,3]
                Adata  = zip(index,values)
                Clarabel.update_A!(solver1,Adata)
                Clarabel.solve!(solver1)

                #new solver 
                A2 = deepcopy(A)
                A2.nzval[index] .= values
                solver2 = Clarabel.Solver(P,q,A2,b,cones,settings)
                Clarabel.solve!(solver2)

                @test isapprox(norm(solver1.solution.x - solver2.solution.x), zero(FloatT), atol=tol)

            end

            @testset "updata q (vector)" begin 
                
                #original problem 
                P,q,A,b,cones,settings = updating_test_data(FloatT)
                solver1   = Clarabel.Solver(P,q,A,b,cones,settings)
                Clarabel.solve!(solver1)

                #change q and re-solve 
                q2 = deepcopy(q)
                q2[1] = 10.

                #revised original solver 
                Clarabel.update_q!(solver1,q2)
                Clarabel.solve!(solver1)

                #new solver 
                solver2 = Clarabel.Solver(P,q2,A,b,cones,settings)
                Clarabel.solve!(solver2)

                @test isapprox(norm(solver1.solution.x - solver2.solution.x), zero(FloatT), atol=tol)

            end

            @testset "updata q (tuple)" begin 
                
                #original problem 
                P,q,A,b,cones,settings = updating_test_data(FloatT)
                solver1   = Clarabel.Solver(P,q,A,b,cones,settings)
                Clarabel.solve!(solver1)

                #revised original solver
                values = FloatT[10.]
                index = [2]
                qdata = zip(index,values)
                Clarabel.update_q!(solver1,qdata)
                Clarabel.solve!(solver1)

                #new solver 
                q2 = deepcopy(q)
                q2[index] .= values
                solver2 = Clarabel.Solver(P,q2,A,b,cones,settings)
                Clarabel.solve!(solver2)

                @test isapprox(norm(solver1.solution.x - solver2.solution.x), zero(FloatT), atol=tol)

            end

            @testset "updata b (vector)" begin 
                
                #original problem 
                P,q,A,b,cones,settings = updating_test_data(FloatT)
                solver1   = Clarabel.Solver(P,q,A,b,cones,settings)
                Clarabel.solve!(solver1)

                #change b and re-solve 
                b2 = deepcopy(b)
                b2 .= 0.

                #revised original solver 
                Clarabel.update_b!(solver1,b2)
                Clarabel.solve!(solver1)

                #new solver 
                solver2 = Clarabel.Solver(P,q,A,b2,cones,settings)
                Clarabel.solve!(solver2)

                @test isapprox(norm(solver1.solution.x - solver2.solution.x), zero(FloatT), atol=tol)

            end

            @testset "updata b (tuple)" begin 
                
                #original problem 
                P,q,A,b,cones,settings = updating_test_data(FloatT)
                solver1   = Clarabel.Solver(P,q,A,b,cones,settings)
                Clarabel.solve!(solver1)

                #revised original solver
                values = FloatT[0., 0.]
                index = [2,4]
                bdata = zip(index,values)
                Clarabel.update_b!(solver1,bdata)
                Clarabel.solve!(solver1)

                #new solver 
                b2 = deepcopy(b)
                b2[index] .= values
                solver2 = Clarabel.Solver(P,q,A,b2,cones,settings)
                Clarabel.solve!(solver2)

                @test isapprox(norm(solver1.solution.x - solver2.solution.x), zero(FloatT), atol=tol)

            end

            @testset "noops" begin 
                
                #original problem 
                P,q,A,b,cones,settings = updating_test_data(FloatT)
                solver   = Clarabel.Solver(P,q,A,b,cones,settings)
                Clarabel.solve!(solver)

                #apply no-op updates to check for crashes 
                Clarabel.update_P!(solver,nothing)
                Clarabel.update_P!(solver,FloatT[])
                Clarabel.update_A!(solver,nothing)
                Clarabel.update_A!(solver,FloatT[])
                Clarabel.update_q!(solver,nothing)
                Clarabel.update_q!(solver,FloatT[])
                Clarabel.update_b!(solver,nothing)
                Clarabel.update_b!(solver,FloatT[])

                #apply global noops 
                Clarabel.update_data!(solver,nothing,nothing,nothing,nothing)
            end

        end      #end "Data Updating Tests (T = $(FloatT))" begin

    end # UnitTestFloats

end #"Basic QP Tests"

nothing
