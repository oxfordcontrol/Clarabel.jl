# -----------------------------------------
# Functions related to `traditional' decomposition
# -----------------------------------------

function decomp_augment_standard!(
  chordal_info::ChordalInfo{T},
  P::SparseMatrixCSC{T},
  q::Vector{T},
  A::SparseMatrixCSC{T},
  b::Vector{T},
) where {T}

  # allocate H and new decomposed cones.  H will be stored
  # in chordal_info and is required for the reversal step 
  H, cones_new = find_standard_H_and_cones!(chordal_info)

  P_new = blockdiag(P, spzeros(T, H.n, H.n))

  q_new = zeros(T, length(q) + H.n)
  q_new[1:length(q)] .= q

  A_new = [A H; spzeros(T, H.n, A.n) -sparse(1.0I, H.n, H.n)]

  b_new = zeros(T, length(b) + H.n)
  b_new[1:length(b)] .= b

  # save the H matrix for use when reconstructing 
  # solution to the original problem
  chordal_info.H = H

  return P_new, q_new, A_new, b_new, cones_new

end 


# Find the transformation matrix `H` and its associated cones for the standard decomposition.

function find_standard_H_and_cones!(
  chordal_info::ChordalInfo{T}, 
) where {T}

  # the cones that we used to form the decomposition 
  cones = chordal_info.init_cones

  # preallocate H and new decomposed cones
  lenH = find_H_col_dimension(chordal_info)
  H_I  = sizehint!(Int[], lenH)

  # ncones from decomposition, plus one for an additional equality constraint
  cones_new = sizehint!(SupportedCone[], final_cone_count(chordal_info) + 1)

  # +1 cone count above is for this equality constraint 
  (_,m) = chordal_info.init_dims
  push!(cones_new, ZeroConeT(m))

  # an iterator for the patterns.  We will expand cones assuming that 
  # they are non-decomposed until we reach an index that agrees with 
  # the internally stored coneidx of the next pattern.   
  patterns_iter = Iterators.Stateful(chordal_info.spatterns)
  row = 1

  for (coneidx,cone) in enumerate(cones)

    if !isempty(patterns_iter) && peek(patterns_iter).orig_index == coneidx
      @assert(isa(cone, PSDTriangleConeT))
      decompose_with_sparsity_pattern!(H_I, cones_new, first(patterns_iter), row)
    else
      decompose_with_cone!(H_I, cones_new, cone, row)
    end

    row += nvars(cone)
  end 

  H = sparse(H_I, collect(1:lenH), ones(T,lenH))

  return H, cones_new
end

function find_H_col_dimension(chordal_info::ChordalInfo{T}) where {T}

  cols, _ = get_decomposed_dim_and_overlaps(chordal_info)
  return cols
end 


function decompose_with_cone!(
  H_I::Vector{Int}, 
  cones_new::Vector{SupportedCone}, 
  cone::SupportedCone, 
  row::Int
)
    for i = 1:nvars(cone)
      push!(H_I,row + i - 1)
    end

    push!(cones_new, deepcopy(cone))
end


function decompose_with_sparsity_pattern!(
  H_I::Vector{Int}, 
  cones_new::Vector{SupportedCone}, 
  spattern::SparsityPattern, 
  row::Int
)
  
  sntree = spattern.sntree

  for i in 1:sntree.n_cliques

    clique = get_clique(sntree, i)

    # the graph and tree algorithms determined the clique vertices of an 
    # AMD-permuted matrix. Since the location of the data hasn't changed 
    # in reality, we have to map the clique vertices back
    c = sort!([spattern.ordering[v] for v in clique])

    add_subblock_map!(H_I, c, row)

    # add a new cone for this subblock
    cdim = get_nblk(sntree, i)
    push!(cones_new, PSDTriangleConeT(cdim))
  end

end


function add_subblock_map!(H_I::Vector{Int}, clique_vertices::Vector{Int}, row_start::Int)

  v = clique_vertices

  for j in eachindex(v), i in 1:j
      row = coord_to_upper_triangular_index((v[i], v[j]))
      push!(H_I, row_start + row - 1)
  end
end


