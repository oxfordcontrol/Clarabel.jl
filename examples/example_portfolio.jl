using LinearAlgebra, SparseArrays, Random, JuMP, Test
using Mosek, MosekTools
using Clarabel
# include("../src\\Clarabel.jl")

## generate the data
rng = Random.MersenneTwister(1)
k = 50; # number of factors
n = k * 100; # number of assets
D = spdiagm(0 => rand(rng, n) .* sqrt(k))
F = sprandn(rng, n, k, 0.5); # factor loading matrix
μ = (3 .+ 9. * rand(rng, n)) / 100. # expected returns between 3% - 12%
γ = 1.0; # risk aversion parameter
d = 1 # we are starting from all cash
x0 = zeros(n);

# ## Transaction costs
# In the model above we assume that trading the assets is free and does not impact the market. However, this is clearly not the case in reality. To make the example more realistic consider the following cost $c_j$ associated with the trade $δ_j = x_j - x_j^0$:
# $$
# c_j(\delta_j) = a_j |\delta_j| + b_j |\delta_j|^{3/2},
# $$
# where the first term models the bid-ask spread and broker fees for asset $j$. The second term models the impact on the market that our trade has. This is obviously only a factor if the volume of our trade is significant. The constant $b_j$ is a function of the total volume traded in the considered time periode and the price volatility of the asset and has to be estimated by the trader. To make this example simple we consider the same coefficients $a$ and $b$ for every asset. The $|\delta_j|^{3/2}$ term can be easily modeled using a power cone constraint $\mathcal{K}_{pow} = \{(x, y, z) \mid x^\alpha y^{(1-\alpha)} \geq |z|, x \geq 0, y \geq 0, 0 \leq \alpha \leq 1 \}$. In fact this can be used to model any market impact function with exponent greater than 1.
# We can write the total transaction cost $a^\top s + b^\top t$ where $s_j$ bounds the absolute value of $\delta_j$ and $t_{j}$ is used to bound the term $|x_j - x_j^0|^{3/2} \leq t_{j}$ using a power cone formulation: $(t_{j}, 1, x_j - x_j^0) \in \mathcal{K}_{pow}(2/3)$.

#-
a = 1e-3
b = 1e-1
γ = 1.0;
# model = JuMP.Model(Mosek.Optimizer)

model = JuMP.Model(Clarabel.Optimizer)

set_optimizer_attribute(model, "direct_solve_method", :qdldl)
set_optimizer_attribute(model, "tol_gap_abs", 1e-8)
set_optimizer_attribute(model, "tol_gap_rel", 1e-8)
# set_optimizer_attribute(model, "min_primaldual_step_length", 1e-1)

@variable(model, x[1:n])
@variable(model, y[1:k])   #this is never used in the model?
@variable(model, s[1:n])
@variable(model, t[1:n])
@objective(model, Min, x' * D * x + y' * y - 1/γ * μ' * x);
@constraint(model, y .== F' * x);
@constraint(model, x .>= 0);

## transaction costs
@constraint(model, sum(x) + a * sum(s) + b * sum(t) == d + sum(x0) );
@constraint(model, [i = 1:n], x[i] - x0[i] <= s[i]); # model the absolute value with slack variable s
@constraint(model, [i = 1:n], x0[i] - x[i] <= s[i]);
@constraint(model, [i = 1:n], [t[i], 1, x[i] - x0[i]] in MOI.PowerCone(2/3));
JuMP.optimize!(model)
# Let's look at the expected return and the total transaction cost:

x_opt = JuMP.value.(x);
y_opt = JuMP.value.(y);
s_opt = JuMP.value.(s);
t_opt = JuMP.value.(t);
expected_return = dot(μ, x_opt)
#-
expected_risk = dot(y_opt, y_opt)
#-
transaction_cost = a * sum(s_opt) + b * sum( t_opt)
