using DataFrames

function Base.show(io::IO, solver::Clarabel.Solver{T}) where {T}
    println(io, "Clarabel model with Float precision: $(T)")
end


# function Base.show(io::IO, settings::Clarabel.Settings{T}) where {T}
#     println("Clarabel settings with Float precision: $(T)")
#     println()
#     for f in fieldnames(Clarabel.Settings)
#         value = getfield(settings,f)
#         vtype = typeof(value)
#         println("$f:$vtype | $value")
#     end
# end


# Inspired by method in https://discourse.julialang.org/t/how-to-align-output-in-columns/3938/2

function Base.show(io::IO, settings::Clarabel.Settings{T}) where {T}

    s = get_precision_string(T)
    println("Clarabel settings with Float precision: $s\n")

    df = DataFrame(Setting = Symbol[], DataType = DataType[], Value = String[])

    for name in fieldnames(Clarabel.Settings)
        value  = getfield(settings,name)
        type   = typeof(value)
        valstr = type == BigFloat ? @sprintf("%g",value) : string(value)
        push!(df, [name, type, valstr])
    end

    strwidths = [maximum(textwidth.(string.([df[:, i]; names(df)[i]]))) for i in 1:size(df, 2)]
    io = IOBuffer()

    # Print headers
    for (i, header) in enumerate(names(df))
        print(io, rpad(header, strwidths[i]), "   ")
    end
    println(io)

    # Print separator
    for (i, header) in enumerate(names(df))
        print(io, "="^strwidths[i], "   ")
    end
    println(io)

    for j in 1:size(df, 1)
        for i in 1:size(df, 2)
            print(io, rpad(df[j,i], strwidths[i]), "   ")
        end
        println(io)
    end

    print(String(take!(io)))
end
