using JSON, SparseArrays, DataStructures

# A struct very similar to the problem data, but containing only
# the data types provided by the user (i.e. no internal types).

mutable struct JsonProblemData{T} 
    P::SparseMatrixCSC{T}
    q::Vector{T}
    A::SparseMatrixCSC{T}
    b::Vector{T}
    cones::Vector{SupportedCone}
    settings::Settings{T}
end



function write_to_file(solver::Solver{T}, file::String) where {T}

    json_data = JsonProblemData(
        deepcopy(solver.data.P),
        deepcopy(solver.data.q),
        deepcopy(solver.data.A),
        deepcopy(solver.data.b),
        deepcopy(solver.data.cones),
        deepcopy(solver.settings),
    )

    dinv = solver.data.equilibration.dinv
    einv = solver.data.equilibration.einv
    c = solver.data.equilibration.c[]

    lrscale!(dinv,json_data.P,dinv)
    json_data.q .*= dinv
    json_data.P .*= inv(c)
    json_data.q .*= inv(c)

    lrscale!(einv,json_data.A,dinv)
    json_data.b .*= einv

    # sanitize settings to remove values that
    # can't be serialized, i.e. infs
    sanitize_settings!(json_data.settings)

    open(file,"w") do io
        JSON.print(io,JSON.lower(json_data))
    end

end


function read_from_file(file::String)

    buffer = open(file, "r") do file
        read(file, String)
    end
    json_data = JSON.parse(buffer)

    P = parse(json_data["P"], SparseMatrixCSC{Float64})
    q = parse(json_data["q"], Vector{Float64})
    A = parse(json_data["A"], SparseMatrixCSC{Float64})
    b = parse(json_data["b"], Vector{Float64})
    cones = parse.(json_data["cones"], SupportedCone)
    settings = parse(json_data["settings"], Settings{Float64})

    # desanitize settings to restore inf bounds 
    desanitize_settings!(settings)

    return Clarabel.Solver(P, q, A, b, cones, settings)

end



function sanitize_settings!(settings::Settings{T}) where T

    for field in fieldnames(typeof(settings))
        value = getfield(settings, field)
        if isa(value,T) && isinf(value)
            setfield!(settings, field, sign(value) * floatmax(T))
        end
    end
end


function desanitize_settings!(settings::Settings{T}) where T

    for field in fieldnames(typeof(settings))
        value = getfield(settings, field)

        if isa(value,T) && floatmax(T) == abs(value)
            setfield!(settings, field, sign(value) * T(Inf))
        end
    end
end




# Julia lowers SparseMatricCSC into a dense-like format, so force 
# it back to a sparse format with the CSC fields 
function JSON.lower(data::JsonProblemData{T}) where T
    return OrderedDict(
        "P" => lower(data.P),
        "q" => data.q,
        "A" => lower(data.A),
        "b" => data.b,
        "cones" => lower.(data.cones),
        "settings" => data.settings,
    )
end


# Julia lowers SparseMatricCSC into a dense-like format, so force 
# it back to a sparse format with the CSC fields 
function lower(A::SparseMatrixCSC{T}) where T
    return OrderedDict(
        "m" => A.m,
        "n" => A.n,
        "colptr" => A.colptr .- 1,
        "rowval" => A.rowval .- 1,
        "nzval" => A.nzval,
    )
end

function lower(cone::SupportedCone)

    #PJG: This won't work on cones with fields other than dim
    typesym = nameof(typeof(cone))

    if isa(cone, PowerConeT)
        return OrderedDict(typesym => cone.α)
    elseif isa(cone,GenPowerConeT)
        return OrderedDict(typesym => [cone.α, cone.dim2])
    else 
        return OrderedDict(typesym => cone.dim)
    end
end


function parse(dict::AbstractDict, ::Type{SparseMatrixCSC{T}}) where{T}
    
    SparseMatrixCSC(
        Int(dict["m"]),
        Int(dict["n"]),
        convert(Vector{Int},dict["colptr"]) .+ 1,
        convert(Vector{Int},dict["rowval"]) .+ 1,
        convert(Vector{T},dict["nzval"]),
    )
end


function parse(data::Vector{Any}, ::Type{Vector{T}}) where T
   T.(data)
end

function parse(dict::AbstractDict, ::Type{Settings{T}}) where T

    settings = Settings{T}()
    for (key, value) in dict

        symbkey = Symbol(key)
        typeval = typeof(getfield(settings, symbkey))

        setfield!(settings, symbkey, typeval(value))
    end
    return settings

 end

 function parse(dict::AbstractDict, ::Type{SupportedCone}) 

    # there should only be 1 key in the dict
    key = collect(keys(dict))[1]
    coneT = eval(Meta.parse(key))

    if key == "GenPowerConeT"
        vals = dict[key]
        α = convert(Vector{Float64}, vals[1])
        dim2 = Int(vals[2])
        return coneT(α,dim2)

    else 
        # all other cones have a single scalar field
        scalar = dict[key]
        return coneT(scalar)
    end

 end