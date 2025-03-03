# wrappers to allow solver configuration calls directly on symbols 
ldlsolver_matrix_shape(x::Symbol) = ldlsolver_matrix_shape(Val{x}())
ldlsolver_constructor(x::Symbol) = ldlsolver_constructor(Val{x}())
