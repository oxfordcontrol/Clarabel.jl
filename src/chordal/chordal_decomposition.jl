
# Chordal decomposition functions.   This code is mostly ported 
# from the original functions in COSMO.jl.

function chordal_decomposition(A::SparseMatrixCSC{T}, b::Vector{T}, cones::Vector{SupportedCone}, settings::Settings{T}) where {T}

    # PJG: not clear if I should return Nothing here or a blank ChordalInfo object or what
    # maybe bail here instead if setting has decomposition disabled, and    
    # then bail after chordal_info is created if no PSD cones are present,
    # returning nothing in both cases 

    # PJG: bail in option disable is internal to this call since that 
    # is how it is handle for the presolver.   Also want to be sure 
    # that return types is consistent with the no_PSD_cones case
    
    # no chordal_info object if decomposition is disabled
    if !settings.chordal_decomposition_enable 
        println("PJG: chordal_decomposition disabled")
        return nothing
    end

    # nothing to do if there are no PSD cones
    if !any(c -> isa(c,PSDTriangleConeT), cones)
        println("PJG: chordal_decomposition no PSD cones")
        return nothing 
    end 

    chordal_info = ChordalInfo(A, b, cones, settings)

    # no decomposition possible 
    if !is_decomposed(chordal_info)
        println("PJG: chordal_decomposition failed")
        return nothing
    end

    return chordal_info
end 
