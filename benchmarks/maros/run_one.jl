using Revise
include(joinpath(@__DIR__,"utils.jl"))
using Clarabel
using Printf
using JuMP
using OSQP, ECOS


file = "HS21.mat"

srcpath = joinpath(@__DIR__,"mat",file)

probdata = matread(srcpath)

solve_ecos(probdata)
#solve_osqp(probdata)
out = solve_clarabel(probdata)

solver = out[2]

return nothing
