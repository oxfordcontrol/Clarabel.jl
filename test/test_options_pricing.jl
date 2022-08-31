# Section 4.4 of
# A. A. Ahmadi, and A. Majumdar
# DSOS and SDSOS Optimization: More Tractable Alternatives to Sum of Squares and Semidefinite Optimization
# 2017

using MultivariateMoments, DynamicPolynomials, JuMP, Clarabel, SumOfSquares


function options_pricing_test(cone::SumOfSquares.PolyJuMP.PolynomialSet,
                              K::Int)

    @polyvar x y z
    σ = [184.04, 164.88, 164.88, 184.04, 164.88, 184.04]
    X = [x^2, x*y, x*z, y^2, y*z, z^2, x, y, z, 1]
    μ = measure([σ .+ 44.21^2; 44.21 * ones(3); 1], X)

    cocone = SumOfSquares.CopositiveInner(cone)

    model = Model(Clarabel.Optimizer)
    @variable(model, p, Poly(X))
    @constraint(model, p in cocone)
    @constraint(model, p - (x - K) in cocone)
    @constraint(model, p - (y - K) in cocone)
    @constraint(model, p - (z - K) in cocone)
    @objective(model, Min, dot(μ, p))

    JuMP.optimize!(model)

end

for K in [40, 45, 50] #[30, 35, 40, 45, 50]
    for cone = [SDSOSCone()]  #[SOSCone(),SDSOSCone(),DSOSCone()]

        println("Cone is type : ", typeof(cone))
        println("K = ", K)

        options_pricing_test(cone,K)
    end
end 


# sos_options_pricing_test(optimizer, config)   = options_pricing_test(optimizer, config, SOSCone(), K, sdsos_cosdsos_exp)
# sd_tests["sos_options_pricing"] = sos_options_pricing_test
# sdsos_options_pricing_test(optimizer, config) = options_pricing_test(optimizer, config, SDSOSCone(), K, sdsos_cosdsos_exp)
# soc_tests["sdsos_options_pricing"] = sdsos_options_pricing_test
# dsos_options_pricing_test(optimizer, config)  = options_pricing_test(optimizer, config, DSOSCone(), K, dsos_codsos_exp)
# linear_tests["dsos_options_pricing"] = dsos_options_pricing_test


#const dsos_codsos_exp   = [132.63, 132.63, 132.63, 132.63, 132.63]
#const sdsos_cosdsos_exp = [ 21.51,  17.17,  13.20,   9.85,   7.30]