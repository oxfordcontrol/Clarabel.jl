
"""
    reverse_decomposition!(ws::COSMO.Workspace, settings::COSMO.Settings)

After the problem has beend solved, undo the chordal decomposition.

Depending on the kind of transformation that was used, this involves:

- Reassembling the original matrix S from its blocks
- Reassembling the dual variable MU and performing a positive semidefinite completion.
reverse_decomposition(data.chordal_info, variables)
"""


#PJG: put these two cases in separate files, and then 
#the top level reverse_decomposition somewhere else.  
# perhaps has a chordal info method.  How does augment work

#PJG : variables here should be "DefaultVariables" only, but this 
# is not enforced since it causes a circular inclusion issue. 

function reverse_decomposition(
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
        reverse_decomposition_compact!(
            chordal_info, new_vars, old_vars, old_cones)
    else
        reverse_decomposition_standard!(
            chordal_info, new_vars, old_vars, old_cones)
    end

    # perform positive semidefinite completion on entries of 
    # z that were not in the decomposed blocks

    #settings.complete_dual && psd_completion!(ws)
    psd_completion!(chordal_info, new_vars)

    return new_vars
end


#-----------------------------------
# reverse the standard decomposition
#-----------------------------------

function reverse_decomposition_standard!(
    chordal_info::ChordalInfo{T},
    new_vars::AbstractVariables{T}, 
    old_vars::AbstractVariables{T},
    _old_cones::Vector{SupportedCone}
) where {T}

    #only H should exist if the standard decomposition was used
    @assert !isnothing(chordal_info.H) && isnothing(chordal_info.cone_maps)

    H     = chordal_info.H 
    (n,m) = variables_dims(new_vars)  

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

# -----------------------------------
# reverse the compact decomposition
# -----------------------------------
  
function reverse_decomposition_compact!(    
    chordal_info::ChordalInfo{T},
    new_vars::AbstractVariables{T}, 
    old_vars::AbstractVariables{T},
    old_cones::Vector{SupportedCone}
) where {T}

    #only cone_map should exist if the compact decomposition was used
    @assert isnothing(chordal_info.H) && !isnothing(chordal_info.cone_maps)

    old_s = old_vars.s
    old_z = old_vars.z
    new_s = new_vars.s
    new_z = new_vars.z

    # the cones for the originating problem, i.e. the cones 
    # that are compatible with the new_vars we want to populate

    cone_maps     = chordal_info.cone_maps
    row_ranges    = _make_rng_conesT(chordal_info.init_cones)

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
            new_s[row_range.start + offset] += old_s[row_ptr + counter]
            # notice: z overwrites (instead of adding) the overlapping entries
            new_z[row_range.start + offset]  = old_z[row_ptr + counter]
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


# -----------------------------------
# psd completion
# -----------------------------------

# # The psd entries of z that correspond to the zeros in s are not constrained by the problem.
# # however, in order to make the dual psd cone positive semidefinite we have to do a
# # positive semidefinite completion routine to choose the values
function psd_completion!(
    chordal_info::ChordalInfo{T}, 
    variables::AbstractVariables{T}
) where {T}

    # working now with the cones from the original
    # problem, not the decomposed ones 
    cones = chordal_info.init_cones

    # loop over psd cones
    row_ranges = _make_rng_conesT(cones)

    # loop over just the patterns 
    for pattern in chordal_info.spatterns
        row_range = row_ranges[pattern.orig_index]
        z = @view variables.z[row_range]
        complete!(z, pattern)
    end

    return nothing
end

function complete!(foo,bar)
    # dummy so I can test without build errors 
end 

# function complete!(μ::AbstractVector{T}, C::PsdConeTriangle{T}, sp_arr::Array{SparsityPattern}, sp_ind::Int, rows::UnitRange{Int}) where {T <: AbstractFloat}
#   sp = sp_arr[sp_ind]

#   μ_view = view(μ, rows)

#   # I want to psd complete y, which is -μ
#   populate_upper_triangle!(C.X, -μ_view, one(T) / sqrt(T(2)))
#   psd_complete!(C.X, C.sqrt_dim, sp.sntree, sp.ordering)
#   extract_upper_triangle!(C.X, μ_view, sqrt(2))
#   @. μ_view *= -one(T)
#   return sp_ind + 1
# end


# positive semidefinite completion (from Vandenberghe - Chordal Graphs..., p. 362)
# input: A - positive definite completable matrix
function psd_complete!(A::AbstractMatrix{T}, pattern::SparsityPattern) where {T <: AbstractFloat}

    sntree  = pattern.sntree
    p  = pattern.ordering
    ip = invperm(p)

    As = Symmetric(A, :U)
    N  = size(As, 1)

    # permutate matrix based on ordering p (p must be a vector type), W is in the order that the cliques are based on
    W = copy(As[p, p])
    W = Matrix(W)

    # go through supernode tree in descending order (given a post-ordering). 
    # This is ensured in the get_snode, get_separators functions
    # PJG: why is this ensured?
    for j = (num_cliques(sntree) - 1):-1:1

        # in order to obtain ν, α the vertex numbers of the supernode are mapped to the new position of the permuted matrix
        # index set of snd(i) sorted using the numerical ordering i,i+1,...i+ni
        ν = get_snode(sntree, j)
        #clique = get_clique(sntree, snd_id)
        # index set containing the elements of col(i) \ snd(i) sorted using numerical ordering σ(i)
        α = get_separators(sntree, j)

        ν = collect(ν) #unborks indexing below
        α = collect(α) #unborks indexing below

        # index set containing the row indices of the lower-triangular zeros in column i (i: representative index) sorted by σ(i)
        i = ν[1]
        η = collect(i+1:1:N)

        # filter out elements in lower triangular part of column i that are non-zero
        filter!(x -> !in(x, α) && !in(x, ν), η)

        Waa = W[α, α]
        Wαν = view(W, α, ν)
        Wηα = view(W, η, α)

        #println("ν  : ", ν)
        #println("α  : ", α)

        #println("Waa: "); display(Waa)
        #println("Wαν: "); display(Wαν)
        #println("Wηα: "); display(Wηα)

        #println("cond(Waa): ", cond(Waa))

        Y = zeros(length(α), length(ν))
        try
            Y[:, :] = Waa \ Wαν
        catch
            #println("Waa is singular")
            Waa_pinv = pinv(Waa)
            Y[:, :] = Waa_pinv * Wαν
        end

        #println("Y: "); display(Y)

    W[η, ν] =  Wηα * Y
    # symmetry condition
    W[ν, η] = view(W, η, ν)'
    end

    # invert the permutation
    A[:, :] =  W[ip, ip]
end
