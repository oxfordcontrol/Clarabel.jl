using LinearAlgebra, SparseArrays
# include("../src\\Clarabel.jl")
using Clarabel
using ConicBenchmarkUtilities

coneMap = Dict(:Zero => Clarabel.ZeroConeT, :Free => :Free,
                        :NonPos => :Nonpositives, :NonNeg => Clarabel.NonnegativeConeT,
                     :SOC => Clarabel.SecondOrderConeT,
                     :ExpPrimal => Clarabel.ExponentialConeT)

filelist = readdir(pwd()*"./primal_exp_cbf")

# dat = readcbfdata("./exp_cbf/car.cbf.gz") # .cbf.gz extension also accepted

for j = 1:32    #length(filelist)
    println("Current file is ", j)
    datadir = filelist[j]   #"gp_dave_1.cbf.gz"
    dat = readcbfdata("./primal_exp_cbf/"*datadir) # .cbf.gz extension also accepted

    println("Current file is: ", datadir)

    # In MathProgBase format:
    c, A, b, con_cones, var_cones, vartypes, sense, objoffset = cbftompb(dat)
    # Note: The sense in MathProgBase form is always minimization, and the objective offset is zero.
    # If sense == :Max, you should flip the sign of c before handing off to a solver.
    if sense == :Max
        c .*= -1
    end

    # T = Float64
    T = BigFloat

    c = T.(c)

    num_con = size(A,1)
    num_var = size(A,2)

    P = spzeros(T,num_var,num_var)

    cone_types = Vector{Clarabel.SupportedCones}(undef, 0)
    cone_dims  = Vector{Int}(undef, 0)
    Anew = Array{T}(undef, 0, num_var)
    bnew = Vector{T}(undef, 0)

    Imatrix = -1.0*Matrix(I, num_var, num_var)

    #Tackling constraint
    for i = 1:length(var_cones)
        cur_var = var_cones[i]
        # println(coneMap[cur_var[1]])

        if coneMap[cur_var[1]] == :Free
            continue
        elseif coneMap[cur_var[1]] == :Nonpositives
            Anew = vcat(Anew, - Imatrix[cur_var[2],:])
            bnew = vcat(bnew, zeros(length(cur_var[2])))

            cone_types = vcat(cone_types, [Clarabel.NonnegativeConeT])
            cone_dims  = vcat(cone_dims, length(cur_var[2]))            
        else 
            Anew = vcat(Anew, Imatrix[cur_var[2],:])
            bnew = vcat(bnew, zeros(length(cur_var[2])))

            cone_types = vcat(cone_types, coneMap[cur_var[1]])
            cone_dims  = vcat(cone_dims, length(cur_var[2]))
        end
    end



    for i = 1:length(con_cones)
        cur_cone = con_cones[i]
        # println(coneMap[cur_cone[1]])

        if coneMap[cur_cone[1]] == :Free
            continue
        elseif coneMap[cur_cone[1]] == :Nonpositives
            Anew = vcat(Anew, - A[cur_cone[2],:])
            bnew = vcat(bnew, -b[cur_cone[2]])

            cone_types = vcat(cone_types, [Clarabel.NonnegativeConeT])
            cone_dims  = vcat(cone_dims, length(cur_cone[2]))               
        else 
            Anew = vcat(Anew, A[cur_cone[2],:])
            bnew = vcat(bnew, b[cur_cone[2]])

            cone_types = vcat(cone_types, coneMap[cur_cone[1]])
            cone_dims  = vcat(cone_dims, length(cur_cone[2]))
        end
    end

    α = Vector{Union{T,Nothing}}(undef, length(cone_dims))
    fill!(α, nothing)

    Anew = sparse(Anew)

    settings = Clarabel.Settings{T}(max_iter=50, direct_solve_method=:qdldl)
    solver   = Clarabel.Solver{T}()
    Clarabel.setup!(solver,P,c,Anew,bnew,cone_types,cone_dims,α,settings)
    Clarabel.solve!(solver)

    GC.gc()
end
