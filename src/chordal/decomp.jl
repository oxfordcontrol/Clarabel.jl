function decomp_augment!(
    chordal_info::ChordalInfo{T}, 
    P::SparseMatrixCSC{T},
    q::Vector{T},
    A::SparseMatrixCSC{T},
    b::Vector{T},
    cones::Vector{SupportedCone},
    settings::Clarabel.Settings{T}
) where{T}

    if settings.chordal_decomposition_compact 
        decomp_augment_compact!(chordal_info, P, q, A, b, cones)
    else
        decomp_augment_standard!(chordal_info, P, q, A, b, cones)
    end

end 


#NB: variables here should be "DefaultVariables" only, but this 
# is not enforced since it causes a circular inclusion issue. 

function decomp_reverse!(
    chordal_info::ChordalInfo{T}, 
    old_vars::AbstractVariables{T}, 
    old_cones::Vector{SupportedCone},
    settings::Settings{T}
) where{T}

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

