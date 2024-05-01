function decomp_augment!(
    chordal_info::ChordalInfo{T}, 
    P::SparseMatrixCSC{T},
    q::Vector{T},
    A::SparseMatrixCSC{T},
    b::Vector{T},
    settings::Settings{T}
) where{T}

    if settings.chordal_decomposition_compact 
        decomp_augment_compact!(chordal_info, P, q, A, b)
    else
        decomp_augment_standard!(chordal_info, P, q, A, b)
    end

end 


#AbstractVariables here should be DefaultVariables only, but this 
# is not enforced since it causes a circular inclusion issue. 

function decomp_reverse!(
    chordal_info::ChordalInfo{T}, 
    old_vars::AbstractVariables{T}, 
    old_cones::Vector{SupportedCone},
    settings::Settings{T}
) where{T}

    # We should have either H (for standard decomp) or cone_maps (for compact decomp)
    # but never both, and they should be consistent with the settings 
    @assert settings.chordal_decomposition_compact == isnothing(chordal_info.H)
    @assert settings.chordal_decomposition_compact != isnothing(chordal_info.cone_maps)

    # here `old_cones' are the ones that were used internally 
    # in the solver, producing internal solution in `old_vars'
    # the cones for the problem as provided by the user or the 
    # upstream presolver are held internally in chordal_info.cones

    (n,m)    = chordal_info.init_dims
    new_vars = typeof(old_vars)(n,m)

    new_vars.x .= old_vars.x[1:n]

    # reassemble the original variables s and z
    if settings.chordal_decomposition_compact
        decomp_reverse_compact!(
            chordal_info, new_vars, old_vars, old_cones)
    else
        decomp_reverse_standard!(
            chordal_info, new_vars, old_vars, old_cones)
    end

    if settings.chordal_decomposition_complete_dual 
        psd_completion!(chordal_info, new_vars)
    end

    return new_vars
end

