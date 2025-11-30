using CSV
using Statistics, LinearAlgebra
using SparseArrays
using Clarabel
using TimerOutputs

include("..//save_table.jl")

function get_return_data(N::Int, rng)
    returns = CSV.File("./sp500_stock_returns_5y.csv")
    returns = returns[rng] # just get first row of returns at time t=0
    R = zeros(length(rng),N)
    companies = propertynames(returns[1])
    for row in 1:lastindex(returns)
        for col in 1:lastindex(companies)-1
            if col <= N
                R[row,col]= getproperty(returns[row],companies[col+1])
            end
        end
    end
    return R
end

"""returns the nxn covariance matrix between the returns of n assets,
stored in R, a matrix of size Txn where T is the number of days in the data (2516)"""
function calc_variance(N::Int, day::Int, R::Matrix{T},rt) where {T}
    Xbar = inv(sqrt(N-1))*(R-ones(day)*rt)

    VarM = Xbar'*Xbar + eps(T)*I      #eps is the regularization ensuring p.s.d.

    return VarM
end

function calc_QR_emd(N::Int, day::Int, R::Matrix{T},rt) where {T}
    Xbar = inv(sqrt(N-1))*(R-ones(day)*rt)
    emd = qr(Xbar)

    return emd
end

function compute_geomean(warm_vec,cold_vec)
    geometric_mean = exp(mean(log.(warm_vec./cold_vec)))

    return geometric_mean
end