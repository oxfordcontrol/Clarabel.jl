using Printf

function save_table(tables,problems,filename,tablename)
    io = open(filename*".tex", "w"); 
    
    println(io,"\\scriptsize") 
    println(io,"\\begin{longtable}" * "{||c" * "||cc"^(2) * "||}") 
    println(io,"\\caption{\\detailtablecaption}") 
    println(io,"\\label{table:"*tablename*"}") 
    println(io,"\\\\") 
    
    #print primary headerss 
    for label in ["iterations", "solve time (s)"] 
        print(io,"& \\multicolumn{2}{c||}{$label}"); 
    end  
    print(io, "\\\\[2ex] \n") 

    #print secondary headers 
    print(io, problems); 
    print(io," & Warm & Cold"^(2)); 
    print(io, "\\\\[1ex]\n") 
    println(io,"\\hline") 
    println(io,"\\endhead") 
    

    for (i,λv) in enumerate(tables[:varVal]) 
        print(io, "\\sc{" * @sprintf("%4.3g",λv)  * "}") 
        iter_warm = tables[:warm_iterations][i] 
        total_time_warm = tables[:warm_time][i] 
        # avg_time_warm = "" 
        iter_cold = tables[:cold_iterations][i] 
        total_time_cold = tables[:cold_time][i] 
        # avg_time_cold = "" 

        if(!isfinite(total_time_warm)) 
            iter_warm = "-" 
            total_time_warm = "-" 
            # avg_time_warm = "-" 
        else  
            total_time_warm = @sprintf("%4.3g",total_time_warm) 
            # avg_time_warm = @sprintf("%4.3g",parse(Float64, total_time_warm)/iter_warm) 
        end 

        if(!isfinite(total_time_cold)) 
            iter_cold = "-" 
            total_time_cold = "-" 
            # avg_time_cold = "-" 
        else  
            total_time_cold = @sprintf("%4.3g",total_time_cold) 
            # avg_time_cold = @sprintf("%4.3g",parse(Float64, total_time_cold)/iter_cold) 
        end 
        print(io, " & $iter_warm & $iter_cold & $total_time_warm & $total_time_cold") 
        print(io, "\\\\ \n") 
    end  

    println(io,"\\end{longtable}") 
    close(io) 
end


##########################################
# Efficient frontier
##########################################
function save_efficientFrontier_table(tables,problems,filename)
    io = open(filename, "w"); 
    
    println(io,"\\scriptsize") 
    println(io,"\\begin{longtable}" * "{||cc" * "||cc"^(2) * "||}") 
    println(io,"\\caption{Efficient frontier}") 
    println(io,"\\label{table:efficient-frontier}") 
    println(io,"\\\\") 
    
    #print primary headerss 
    print(io,"\\multicolumn{2}{||c||}{{Values}} ")
    for label in ["iterations", "solve time (s)"] 
        print(io,"& \\multicolumn{2}{c||}{{$label}}"); 
    end  
    print(io, "\\\\[2ex] \n") 
    println(io,"\\hline") 

    #print secondary headers 
    print(io, " \$r_0 \$ & \$f(r_0)\$"); 
    print(io," & Warm & Cold"^(2)); 
    print(io, "\\\\[1ex]\n") 
    println(io,"\\hline") 
    println(io,"\\endhead") 
    

    for (i,tval) in enumerate(tables[:t]) 
        print(io, "\\sc{" * @sprintf("%4.3g",tval)  * "}") 
        print(io, " & \\sc{" * @sprintf("%4.3g",tables[:ft][i])  * "}") 
        iter_warm = tables[:warm_iterations][i] 
        total_time_warm = tables[:warm_time][i] 
        # avg_time_warm = "" 
        iter_cold = tables[:cold_iterations][i] 
        total_time_cold = tables[:cold_time][i] 
        # avg_time_cold = "" 

        if(!isfinite(total_time_warm)) 
            iter_warm = "-" 
            total_time_warm = "-" 
            # avg_time_warm = "-" 
        else  
            total_time_warm = @sprintf("%4.3g",total_time_warm) 
            # avg_time_warm = @sprintf("%4.3g",parse(Float64, total_time_warm)/iter_warm) 
        end 

        if(!isfinite(total_time_cold)) 
            iter_cold = "-" 
            total_time_cold = "-" 
            # avg_time_cold = "-" 
        else  
            total_time_cold = @sprintf("%4.3g",total_time_cold) 
            # avg_time_cold = @sprintf("%4.3g",parse(Float64, total_time_cold)/iter_cold) 
        end 
        print(io, " & $iter_warm & $iter_cold & $total_time_warm & $total_time_cold") 
        print(io, "\\\\ \n") 
    end  

    println(io,"\\end{longtable}") 
    close(io) 
end

