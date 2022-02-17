
using Revise
include("../IPSolver.jl")
using .IPSolver
using LinearAlgebra
using Printf
using MAT
using JuMP
using OSQP, ECOS
include("./utils.jl")


function print_header()

    println("                             ECOS                             CLARABEL          ")
    println("                  TIME       STAT       COST   |     TIME       STAT       COST      ")
    println("------------------------------------------------------------------------------------------------")

end

function print_row(i,name,result_ecos,result_clarabel)

    @printf("%3i %-9s : ", i, name)

    for result in [result_ecos,result_clarabel]
        stat = @sprintf("%s",result.status)[1:6]
        @printf("%+9.2e   %-4s    %+9.2e  : ", result.time, stat, result.cost)
    end
    @printf("\n")

end



srcpath = joinpath(@__DIR__,"mat")
#get Maros archive path and get names of data files
files = filter(endswith(".mat"), readdir(srcpath))


result_ecos = []
result_osqp = []
result_clarabel = []
names = []

#This file indices cause segfaults in JuMP (not ECOS).  Just skip them
badfiles = [78 79 117]

for FNUM = 1:length(files) #length(files)

    if(any(FNUM .== badfiles))
        continue
    end

    push!(names,files[FNUM][1:end-4])

    println("SOLVING PROBLEM ", names[end], " FILE NUMBER, ", FNUM)

    #load problem data
    println("Loading file")
    thisfile = joinpath(srcpath,files[FNUM])
    probdata = matread(thisfile)

    push!(result_ecos, solve_ecos(probdata))

    # push!(result_osqp, solve_osqp(probdata))

    push!(result_clarabel,solve_clarabel(probdata))

end

print_header()
for i = 1:length(result_ecos)
    print_row(i, names[i], result_ecos[i],result_clarabel[i])
end
