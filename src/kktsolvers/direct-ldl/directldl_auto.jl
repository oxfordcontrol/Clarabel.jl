struct AutoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    function AutoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        autokey = get_auto_ldl_solver()
        ldlsolver_constructor(autokey){T}(KKT,Dsigns,settings)
    end
end


ldlsolver_constructor(::Val{:auto}) = AutoDirectLDLSolver
ldlsolver_matrix_shape(::Val{:auto}) = ldlsolver_matrix_shape(get_auto_ldl_solver())
ldlsolver_is_available(::Val{:auto}) = true

function get_auto_ldl_solver()

    # priority in Panua-Pardiso, MKL-Pardiso, HSL, QDLDL.   All others 
    # must be manually selected 

    priority = [:panua,:mkl,:ma57,:qdldl]

    for key in priority
        if ldlsolver_is_available(key)
            return key
        end
    end

end 

