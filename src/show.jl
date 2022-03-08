

function Base.show(io::IO, solver::Clarabel.Solver{T}) where {T}
    println(io, "Clarabel model with Float precision: $(T)")
end


function Base.show(io::IO, settings::Clarabel.Settings{T}) where {T}
    dump(settings)
end
