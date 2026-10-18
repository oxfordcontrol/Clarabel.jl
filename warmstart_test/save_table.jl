using Printf
include("comparison_utils.jl")

function _comparison_columns(tables)
    return haskey(tables,:convex_time) ?
        ((:warm,"Smooth"),(:convex,"Convex"),(:cold,"Cold")) :
        ((:warm,"Warm"),(:cold,"Cold"))
end

function _save_comparison_table(tables,filename,tablename,caption,headers,values)
    columns = _comparison_columns(tables)
    nmethods = length(columns)
    mkpath(dirname(filename))
    open(filename,"w") do io
        println(io,"\\scriptsize")
        println(io,"\\begin{longtable}{||",repeat("c",length(headers)),
                "||",repeat("c",nmethods),"||",repeat("c",nmethods),"||}")
        println(io,"\\caption{",caption,"}")
        println(io,"\\label{table:",tablename,"}\\\\")
        print(io,repeat(" & ",length(headers)-1))
        for label in ("iterations","solve time (s)")
            print(io," & \\multicolumn{",nmethods,"}{c||}{",label,"}")
        end
        println(io,"\\\\[2ex]")
        print(io,join(headers," & "))
        for _ in 1:2, (_,label) in columns
            print(io," & ",label)
        end
        println(io,"\\\\[1ex]\n\\hline\n\\endhead")
        for i in eachindex(first(values))
            print(io,join([@sprintf("%4.3g",v[i]) for v in values]," & "))
            for (method,_) in columns
                success = isfinite(tables[Symbol(method,:_time)][i])
                print(io," & ",success ? string(tables[Symbol(method,:_iterations)][i]) : "-")
            end
            for (method,_) in columns
                time = tables[Symbol(method,:_time)][i]
                print(io," & ",isfinite(time) ? @sprintf("%4.3g",time) : "-")
            end
            println(io,"\\\\")
        end
        println(io,"\\end{longtable}")
    end
    return nothing
end

function save_table(tables,problems,filename,tablename)
    return _save_comparison_table(tables,filename*".tex",tablename,
        "\\detailtablecaption",[problems],[tables[:varVal]])
end

function save_efficientFrontier_table(tables,problems,filename)
    return _save_comparison_table(tables,filename*".tex","efficient-frontier-comparison",
        "Efficient frontier",[problems,raw"$f(r_0)$"],[tables[:t],tables[:ft]])
end
