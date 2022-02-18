using Revise
include("./utils.jl")
include("../../IPSolver.jl")
using .IPSolver
using Printf
using JuMP
using OSQP, ECOS


file = "HS53.mat"

srcpath = joinpath(@__DIR__,"mat",file)

probdata = matread(srcpath)

solve_ecos(probdata)
#solve_osqp(probdata)
out = solve_clarabel(probdata)

solver = out[2]

return nothing
