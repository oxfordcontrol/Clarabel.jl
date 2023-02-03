#=
list of predefined signomials and domains from various applications

prefixes:
- CS16 refers to
"Relative Entropy Relaxations for Signomial Optimization"
(2016) by Chandrasekaran & Shah
- MCW19 refers to
"Signomial and Polynomial Optimization via Relative Entropy and Partial Dualization"
(2019) by Murray, Chandrasekaran, & Wierman

c are coefficients
A are powers
x is a feasible point in domain, if known
obj_ub is an objective upper bound (value at feasible point), if known
=#

using LinearAlgebra
import Distributions
import Random
using SparseArrays

signomialmin_data = Dict{Symbol, NamedTuple}(
    :motzkin2 => (
        # f = 1 - 3*x1^2*x2^2 + x1^2*x2^4 + x1^4*x2^2
        fc = [1, -3, 1, 1],
        fA = [0 0; 2 2; 2 4; 4 2],
        gc = [],
        gA = [],
        obj_ub = 0.0,
        ),
    :motzkin3 => (
        # f = x3^6 - 3*x1^2*x2^2*x3^2 + x1^2*x2^4 + x1^4*x2^2
        fc = [0, 1, -3, 1, 1],
        fA = [0 0 0; 0 0 6; 2 2 2; 2 4 0; 4 2 0],
        gc = [],
        gA = [],
        obj_ub = 0.0,
        ),
    :CS16ex8_13 => (
        fc = [0, 10, 10, 10, -14.6794, -7.8601, 8.7838],
        fA = [0 0 0; 10.2 0 0; 0 9.8 0; 0 0 8.2; 1.5089 1.0981 1.3419;
            1.0857 1.9069 1.6192; 1.0459 0.0492 1.6245],
        gc = [],
        gA = [],
        obj_ub = -0.9748,
        ),
    :CS16ex8_14 => (
        fc = [0, 10, 10, 10, 7.5907, -10.9888, -13.9164],
        fA = [0 0 0; 10.2 0 0; 0 9.8 0; 0 0 8.2; 1.9864 0.2010 1.0855;
            2.8242 1.9355 2.0503; 0.1828 2.7772 1.9001],
        gc = [],
        gA = [],
        obj_ub = -0.739,
        ),
    :CS16ex18 => (
        fc = [0, 10, 10, 10, -14.6794, -7.8601, 8.7838],
        fA = [0 0 0; 10.2070 0.0082 -0.0039; -0.0081 9.8024 -0.0097;
            0.0070 -0.0156 8.1923; 1.5296 1.0927 1.3441;
            1.0750 1.9108 1.6339; 1.0513 0.0571 1.6188],
        gc = [],
        gA = [],
        obj_ub = -0.9441,
        ),
    :CS16ex12 => (
        fc = [0, 10, 10, 10, -14.6794, -7.8601, 8.7838],
        fA = [0 0 0; 10.2 0 0; 0 9.8 0; 0 0 8.2; 1.5089 1.0981 1.3419;
            1.0857 1.9069 1.6192; 1.0459 0.0492 1.6245],
        gc = [[1, -8, -8, -8, -6.4]],
        gA = [[0 0 0; 10.2 0 0; 0 9.8 0; 0 0 8.2; 1.0857 1.9069 1.6192]],
        obj_ub = -0.6144,
        ),
    :CS16ex13 => (
        fc = [0, 10, 10, 10, -14.6794, -7.8601, 8.7838],
        fA = [0 0 0; 10.2 0 0; 0 9.8 0; 0 0 8.2; 1.5089 1.0981 1.3419;
            1.0857 1.9069 1.6192; 1.0459 0.0492 1.6245],
        gc = [[-8, -8, -8, 0.7410, -0.4492, 1.4240]],
        gA = [[10.2 0 0; 0 9.8 0; 0 0 8.2; 1.5089 1.0981 1.3419;
            1.0857 1.9069 1.6192; 1.0459 0.0492 1.6245]],
        obj_ub = -0.7372,
        ),
    :MCW19ex1_mod => (
        fc = [0, 0.5, -1, 0],
        fA = [0 0 0; 1 -1 0; 1 0 0; 0 -1 0],
        gc = [[100, -1, -1, -0.05], [-70, 1], [-1, 1], [-0.5, 1], [150, -1],
            [30, -1], [21, -1]],
        gA = [[0 0 0; 0 1 -1; 0 1 0; 1 0 1], [0 0 0; 1 0 0], [0 0 0; 0 1 0],
            [0 0 0; 0 0 1], [0 0 0; 1 0 0], [0 0 0; 0 1 0], [0 0 0; 0 0 1]],
        obj_ub = -147.5,
        ),
    :MCW19ex8 => (
        fc = [0, 0.05, 0.05, 0.05, 1],
        fA = sparse([2, 3, 4, 5], [1, 2, 3, 9], [1, 1, 1, 1], 5, 10),
        gc = [[1, 0.5, -1], [1, 0.5, -1], [1, 0.5, -1], [1, -0.25, -0.5],
            [1, -0.79681], [1, -0.79681], [1, -0.79681]],
        gA = [
            sparse([2, 2, 2, 3, 3], [1, 4, 7, 10, 7], [1, 1, -1, 1, -1], 3, 10),
            sparse([2, 2, 2, 3, 3], [2, 5, 8, 7, 8], [1, 1, -1, 1, -1], 3, 10),
            sparse([2, 2, 2, 3, 3], [3, 6, 9, 8, 9], [1, 1, -1, 1, -1], 3, 10),
            sparse([2, 3, 3], [10, 9, 10], [-1, 1, -1], 3, 10),
            sparse([2, 2], [4, 7], [1, -1], 2, 10),
            sparse([2, 2], [5, 8], [1, -1], 2, 10),
            sparse([2, 2], [6, 9], [1, -1], 2, 10),
            ],
        obj_ub = 0.2056534,
        ),
    )

function signomialmin_random(
    m::Int,
    n::Int;
    num_samples::Int = 100,
    neg_c_frac::Real = 0.0,
    sparsity::Real = 0.3,
    rseed::Int = 1,
    )
    Random.seed!(rseed)

    # generate random objective signomial
    fc = rand(m)
    for k in 1:m
        if rand() < neg_c_frac
            fc[k] *= -1
        end
    end
    fA = vcat(zeros(1, n), Matrix(sprandn(m - 1, n, sparsity)))
    for k in 2:size(fA, 1)
        if iszero(norm(fA[k, :]))
            fA[k, :] = randn(n)
        end
    end

    # bounded domain set (in exp space) is intersection of positives and ball
    gc = [vcat(1, fill(-1, n))]
    gA = [sparse(2:(n + 1), 1:n, fill(2, n))]

    # sample points to get an objective upper bound
    eval_signomial(c::Vector, A::AbstractMatrix, x::Vector) =
        sum(c_k * exp(dot(A_k, x)) for (c_k, A_k) in zip(c, eachrow(A)))
    obj_ub = Inf
    for i in 1:num_samples
        x = abs.(randn(n))
        r = rand(Distributions.Exponential(0.5))
        x /= sqrt(sum(abs2, x) + r)
        x = log.(x)
        @assert eval_signomial(gc[1], gA[1], x) >= 0
        obj_ub = min(obj_ub, eval_signomial(fc, fA, x))
    end

    return (fc, fA, gc, gA, obj_ub)
end
