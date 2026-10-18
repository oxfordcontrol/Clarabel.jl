using Clarabel, LinearAlgebra, SparseArrays, JLD2, Test
include("save_table.jl")

# A small repeated SOCP, requiring neither MNIST nor the stock/benchmark data.
P = sparse(Matrix{Float64}(I,3,3))
A = -P
b = zeros(3)
settings = Clarabel.Settings(verbose=false,presolve_enable=false,
                             chordal_decomposition_enable=false)
initial = Clarabel.Solver(P,[-1.0,-2.0,0.0],A,b,
    Clarabel.SupportedCone[Clarabel.SecondOrderConeT(3)],settings)
Clarabel.solve!(initial)
@test initial.solution.status == Clarabel.SOLVED
warmup_comparison!(initial)
smooth,convex,cold = deepcopy(initial),deepcopy(initial),deepcopy(initial)
values = [2.1,2.2,2.3]
tables = Dict{Symbol,Any}(:varVal=>values,:warm_time=>fill(Inf,3),
    :cold_time=>fill(Inf,3),:warm_iterations=>fill(-1,3),:cold_iterations=>fill(-1,3))
init_convex_comparison!(tables,3)
for (i,value) in enumerate(values)
    q = [-1.0,-value,0.0]
    for solver in (smooth,convex,cold)
        Clarabel.update_q!(solver,q)
    end
    Clarabel.solve!(smooth; warmstart=true)
    record_comparison_result!(tables,:warm,smooth,i)
    solve_convex_comparison!(convex,tables,i)
    Clarabel.solve!(cold)
    record_comparison_result!(tables,:cold,cold,i)
    @test smooth.solution.x ≈ cold.solution.x atol=1e-6
    @test convex.solution.x ≈ cold.solution.x atol=1e-6
end
for method in (:warm,:convex,:cold)
    @test all(==("SOLVED"),tables[Symbol(method,:_status)])
end

output = haskey(ENV,"CLARABEL_COMPARISON_OUTPUT") ?
    ENV["CLARABEL_COMPARISON_OUTPUT"] : mktempdir(; cleanup=false)
filename = joinpath(output,"smoke")
path = save_comparison(tables,filename)
saved = JLD2.load(path)
@test saved["methods"] == ["smoothing","convex_combination","cold"]
@test saved["convex_weight"] == 0.01
@test saved["iter_cc"] == tables[:convex_iterations]
@test saved["solve_time_cc"] == tables[:convex_time]
save_table(tables,"Parameter",filename,"smoke-comparison")
@test isfile(filename*".tex")
println("Three-method smoke comparison passed. Results: ",path)
