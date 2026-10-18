using Clarabel, JLD2

const PAPER_CASES = (
    "svm_hyperparameter_tuning", "svm_robust_hyperparameter_tuning",
    "portfolio", "portfolio_efficientFrontier",
)

json_safe(x::AbstractDict) = Dict(string(k)=>json_safe(v) for (k,v) in x)
json_safe(x::AbstractArray) = map(json_safe,x)
json_safe(x::AbstractFloat) = isfinite(x) ? x : nothing
json_safe(x) = x

results = Dict{String,Any}()
for name in PAPER_CASES
    path = joinpath(@__DIR__,"results",name*"_comparison.jld2")
    data = JLD2.load(path)
    n = length(data["cold_iterations"])
    for method in ("warm","convex","cold")
        length(data[method*"_status"]) == n || error("Incomplete status data: $name")
        any(==("NOT_RUN"),data[method*"_status"]) && error("Unfinished experiment: $name")
    end
    completed = [i for i in 1:n if all(data[m*"_status"][i] == "SOLVED"
        for m in ("warm","convex","cold"))]
    data["max_objective_spread"] = isempty(completed) ? NaN : maximum(
        maximum(data[m*"_objective"][i] for m in ("warm","convex","cold")) -
        minimum(data[m*"_objective"][i] for m in ("warm","convex","cold"))
        for i in completed
    )
    results[name] = json_safe(data)
    println(name,": ",join([m*"="*string(data[m*"_success_count"])*"/"*string(n)
        for m in ("warm","convex","cold")],", "),
        "; maximum objective spread = ",data["max_objective_spread"])
end
output = joinpath(@__DIR__,"results","paper_comparison_results.json")
open(output,"w") do io
    Clarabel.JSON.print(io,results,2)
end
println("Exported ",output)
