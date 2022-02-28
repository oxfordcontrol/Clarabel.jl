
#using Revise
include(joinpath(@__DIR__,"../../IPSolver.jl"))
include(joinpath(@__DIR__,"utils.jl"))
using LinearAlgebra
using Printf
using MAT
using JuMP
using OSQP, ECOS
using JLD2



function print_header()

    println("                             ECOS                             CLARABEL          ")
    println("                  TIME       STAT       COST      |   TIME       STAT       COST      |   REFOBJ     SPEEDUP ")
    println("------------------------------------------------------------------------------------------------")

end

function print_row(i,name,result_ecos,result_clarabel, ref_sol)

    @printf("%3d %-9s : ", i, name)

    for result in [result_ecos,result_clarabel]
        stat = @sprintf("%s",result.status)[1:6]
        @printf("%+9.2e   %-4s    %+10.3e  : ", result.time, stat, result.cost)
    end
    speedup = result_ecos.time / result_clarabel.time
    @printf("%+9.3e  ", ref_sol)
    @printf("%+9.3f  ", speedup)
    @printf("\n")

end



srcpath = joinpath(@__DIR__,"mat")
#get Maros archive path and get names of data files
files = filter(endswith(".mat"), readdir(srcpath))


result_ecos = []
result_osqp = []
result_clarabel = []
names = []

#These file indices cause segfaults in JuMP (not ECOS).  Just skip them
#badfiles = [78, 79, 117]
#push!(badfiles,36)   #this is EXDATA, which is huge

solve_list = 1:length(files)

for FNUM = solve_list #length(files)

    #if(any(FNUM .== badfiles))
    #    continue
    #end

    push!(names,files[FNUM][1:end-4])

    println("SOLVING PROBLEM ", names[end], " FILE NUMBER, ", FNUM)

    #load problem data
    println("Loading file")
    thisfile = joinpath(srcpath,files[FNUM])
    probdata = matread(thisfile)

    push!(result_ecos, solve_ecos_eq(probdata))
    println("Result ECOS: ", result_ecos[end].status)

    # push!(result_osqp, solve_osqp(probdata))

    push!(result_clarabel,solve_clarabel_eq(probdata)[1])
    println("Result Clarabel: ", result_clarabel[end].status)

end

#Note objective function source file doesn't
#include underscores in problem names
refsols = get_ref_solutions(joinpath(@__DIR__,"ref_solutions.txt"))
no_underscore = x -> replace(x,"_" => "")

print_header()
for i = 1:length(result_ecos)
    objname = no_underscore(names[i])
    print_row(i, names[i], result_ecos[i],result_clarabel[i], refsols[objname])

end

jldsave("maros_results_tmp0_maxiter100.jld2";names,result_ecos,result_clarabel,refsols)
