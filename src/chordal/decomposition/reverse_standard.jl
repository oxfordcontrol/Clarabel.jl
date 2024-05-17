#-----------------------------------
# reverse the standard decomposition
#-----------------------------------

function decomp_reverse_standard!(
    chordal_info::ChordalInfo{T},
    new_vars::AbstractVariables{T}, 
    old_vars::AbstractVariables{T},
    _old_cones::Vector{SupportedCone}
) where {T}

    H     = chordal_info.H 
    (_,m) = variables_dims(new_vars)  

    mul!(new_vars.s, H, @view old_vars.s[(1+m):end])
    mul!(new_vars.z, H, @view old_vars.z[(1+m):end])

    # to remove the overlaps we take the average of the values for
    # each overlap by dividing by the number of blocks that overlap
    #in a particular entry, i.e. number of 1s in each row of H

    rows, nnzs = number_of_overlaps_in_rows(H)

    for (ri, nnz) in zip(rows,nnzs)
        new_vars.z[ri] /= nnz
    end

end

function number_of_overlaps_in_rows(
    A::SparseMatrixCSC{T}
) where {T}

    # sum the entries row-wise
    n_overlaps = sum(A, dims = 2)
    ri = findall(x -> x > 1, n_overlaps)
    return ri, n_overlaps[ri]

  end