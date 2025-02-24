struct AutoDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    function AutoDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        autokey = get_auto_ldl_solver()
        Clarabel.DirectLDLSolversDict[autokey]{T}(KKT,Dsigns,settings)
    end
end

DirectLDLSolversDict[:auto] = AutoDirectLDLSolver


function required_matrix_shape(::Type{AutoDirectLDLSolver}) 
    autokey = get_auto_ldl_solver()
    required_matrix_shape(Clarabel.DirectLDLSolversDict[autokey])
end 



function get_auto_ldl_solver()

    # priority in Panua-Pardiso, MKL-Pardiso, HSL, QDLDL.   All others 
    # must be manually selected 

    if haskey(Clarabel.DirectLDLSolversDict,:panua) && Clarabel.Pardiso.panua_is_available()
        return :panua
    end

    if haskey(Clarabel.DirectLDLSolversDict,:mkl) && Clarabel.Pardiso.mkl_is_available()
        return :mkl
    end

    if haskey(Clarabel.DirectLDLSolversDict,:ma57)
        return :ma57
    end

    return :qdldl

end 

