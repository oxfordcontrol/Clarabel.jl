using Revise
include("./utils.jl")
include("../../IPSolver.jl")
using .IPSolver
using Printf
using JuMP
using OSQP, ECOS


file = "QSCAGR25.mat"

srcpath = joinpath(@__DIR__,"mat",file)

probdata = matread(srcpath)

solve_ecos(probdata)
out = solve_clarabel(probdata)

solver = out[2]

return nothing
