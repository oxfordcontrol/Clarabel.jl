#AbstractVariables here should be DefaultVariables only, but this 
# is not enforced since it causes a circular inclusion issue. 

# -----------------------------------
# reverse the compact decomposition
# -----------------------------------
  
function decomp_reverse_compact!(    
    chordal_info::ChordalInfo{T},
    new_vars::AbstractVariables{T}, 
    old_vars::AbstractVariables{T},
    old_cones::Vector{SupportedCone}
) where {T}

    old_s = old_vars.s
    old_z = old_vars.z
    new_s = new_vars.s
    new_z = new_vars.z

    # the cones for the originating problem, i.e. the cones 
    # that are compatible with the new_vars we want to populate,
    # are held in chordal_info.init_cones

    cone_maps     = chordal_info.cone_maps
    row_ranges    = collect(rng_cones_iterator(chordal_info.init_cones))

    row_ptr = 1 

    for (cone, cone_map) in zip(old_cones,cone_maps)

        row_range = row_ranges[cone_map.orig_index]

        if isnothing(cone_map.tree_and_clique)
            row_ptr = add_blocks_with_cone!(
                new_s, old_s, new_z, old_z, row_range, cone, row_ptr)
        
            else
            @assert isa(cone, PSDTriangleConeT)
            @assert !isnothing(cone_map.tree_and_clique)

            (tree_index, clique_index) = cone_map.tree_and_clique
            pattern   = chordal_info.spatterns[tree_index]

            row_ptr = add_blocks_with_sparsity_pattern!(
                new_s, old_s, new_z, old_z, row_range, pattern, clique_index, row_ptr)
        end
    end 

end 

function add_blocks_with_sparsity_pattern!(
    new_s::Vector{T}, 
    old_s::Vector{T}, 
    new_z::Vector{T}, 
    old_z::Vector{T}, 
    row_range::UnitRange{Int}, 
    spattern::SparsityPattern, 
    clique_index::Int,
    row_ptr::Int
) where {T}

    sntree   = spattern.sntree
    ordering = spattern.ordering

    clique  = sort!([ordering[v] for v in get_clique(sntree, clique_index)])
    counter = 0
    for j in clique, i in clique
        if i <= j
            offset = coord_to_upper_triangular_index((i, j)) - 1
            @views new_s[row_range.start + offset] += old_s[row_ptr + counter]
            # notice: z overwrites (instead of adding) the overlapping entries
            @views new_z[row_range.start + offset]  = old_z[row_ptr + counter]
            counter += 1
        end
    end
    
    row_ptr + triangular_number(length(clique))
end
    
function add_blocks_with_cone!(
    new_s::Vector{T}, 
    old_s::Vector{T}, 
    new_z::Vector{T}, 
    old_z::Vector{T}, 
    row_range::UnitRange{Int}, 
    cone::SupportedCone, 
    row_ptr::Int
) where {T}

    src_range = row_ptr:(row_ptr + nvars(cone) - 1)
    @views new_s[row_range] .= old_s[src_range]
    @views new_z[row_range] .= old_z[src_range]
    row_ptr += nvars(cone)

end 
