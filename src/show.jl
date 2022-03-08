

function Base.show(io::IO, solver::IPSolver.Solver{T}) where {T}
    println(io, "IPSolver model with Float precision: $(T)")
end


function Base.show(io::IO, settings::IPSolver.Settings{T}) where {T}
    dump(settings)
end
