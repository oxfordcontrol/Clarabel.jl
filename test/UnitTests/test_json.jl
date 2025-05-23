using Test, LinearAlgebra, SparseArrays, Clarabel
@testset "json" begin

    P = sparse([4. 1;1 2])
    c = [1; 1.]
    A = sparse([1. 1;1 0; 0 1])
    b = [1.;1;1]

    cones = [Clarabel.NonnegativeConeT(1), 
            Clarabel.ZeroConeT(1), 
            Clarabel.NonnegativeConeT(1)]

    settings = Clarabel.Settings()
    solver = Clarabel.Solver(P,c,A,b,cones)

    # write to a JSON file then reload 
    file = tempname() * ".json"
    Clarabel.save_to_file(solver, file)
    solver2 = Clarabel.load_from_file(file)

    # solve both problems and compare
    Clarabel.solve!(solver)
    Clarabel.solve!(solver2)
    @test isapprox(solver.solution.x, solver2.solution.x, atol=1e-10)
    @test isequal(solver.solution.status, solver2.solution.status)


    # test loading with custom settings 
    settings = Clarabel.Settings()
    settings.max_iter = 1
    solver3 = Clarabel.load_from_file(file,settings)
    Clarabel.solve!(solver3)
    @test isequal(solver3.solution.status,Clarabel.MAX_ITERATIONS)

end
nothing
