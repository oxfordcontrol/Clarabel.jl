# wrappers to allow solver configuration calls directly on symbols 
ldlsolver_matrix_shape(x::Symbol) = ldlsolver_matrix_shape(Val{x}())
ldlsolver_constructor(x::Symbol) = ldlsolver_constructor(Val{x}())
ldlsolver_is_available(x::Symbol) = ldlsolver_is_available(Val{x}())

function ldlsolver_default_error(x::Symbol)
    if x ∈ [:panua,:mkl]
        return "$x is not available.  Have you loaded Pardiso.jl?"
    elseif x ∈ [:ma57]
        return "$x is not available.  Have you loaded HSL.jl?"
    else
        return "No solver found for option $x"
    end
end 