# -----------------------------------------
# Functions related to "traditional" decomposition
# -----------------------------------------

function augment_standard!(
  chordal_info::ChordalInfo{T},
  P::SparseMatrixCSC{T},
  q::Vector{T},
  A::SparseMatrixCSC{T},
  b::Vector{T},
  cones::Vector{SupportedCone}
) where {T}

  # allocate H and new decomposed cones.  H is stored
  # in chordal_info and required for transformation 
  # of the augmented solution back to the original one 
  # for the "compact" transformation, H = nothing 

  # PJG: why doesn't chordal_info already know about the initial cones?
  # not needed as an argument here?
  # PJG: would prefer to use the internal cones only and 
  # only take P,q,A,b as arguments 
  H, cones_new = find_standard_H_and_cones!(chordal_info, cones)

  z = zeros(T,H.n)

  P_new = blockdiag(P, spzeros(H.n, H.n))
  q_new = [q; z]
  A_new = [A H; spzeros(H.n, A.n) -sparse(1.0I, H.n, H.n)]
  b_new = [b; z]

  # save the decomposition matrix for use when 
  # reconstructing the solution 
  chordal_info.H = H

  return P_new, q_new, A_new, b_new, cones_new

end 



" Find the transformation matrix `H` and its associated cones for the standard decomposition."

function find_standard_H_and_cones!(
  chordal_info::ChordalInfo{T}, 
  cones::Vector{SupportedCone}
) where {T}

  # preallocate H and new decomposed cones
  n = find_H_col_dimension(chordal_info)

  H_I       = sizehint!(Int[], n)

  # ncones from decomposition, plus one for an additional equality constraint
  cones_new = sizehint!(SupportedCone[], post_cone_count(chordal_info) + 1)

  # +1 cone count above is for this equality constraint 
  push!(cones_new, ZeroConeT(chordal_info.init_dims[1]))

  # an iterator for the patterns.  We will expand cones assuming that 
  # they are non-decomposed until we reach an index that agrees with 
  # the internally stored coneidx of the next pattern.   
  patterns_iter = Iterators.Stateful(chordal_info.spatterns)
  row = 1

  for (coneidx,cone) in enumerate(cones)

    if !isempty(patterns_iter) && peek(patterns_iter).coneidx == coneidx

      @assert(isa(cone, PSDTriangleConeT))
      decompose_with_sparsity_pattern!(H_I, cones_new, first(patterns_iter), row)

    else
      decompose_with_cone!(H_I, cones_new, cone, row)

    end

    row += nvars(cone)
  end 

  H = sparse(H_I, collect(1:n), ones(n))

  return H, cones_new
end

function decompose_with_cone!(H_I::Vector{Int}, cones_new::Vector{SupportedCone}, cone::SupportedCone, row::Int)
  
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

  for i in 1:num_cliques(sntree)

    clique = get_clique(sntree, i)

    # the graph and tree algorithms determined the clique vertices of an 
    # AMD-permuted matrix. Since the location of the data hasn't changed 
    # in reality, we have to map the clique vertices back
    c = sort!([spattern.ordering[v] for v in clique])

    add_subblock_map!(H_I, c, row)

    # PJG: compact transform seems to maintain cone_map roughly here as well
    # compact transform also tags the new cones with their tree and clique 
    # number in add_entries!    I think it would be preferable to store all 
    # of the origin data, tree and clique number into the cone_map, and keep 
    # it a common data object between the transformations 

    # add a new cone for this subblock
    cdim = get_nblk(sntree, i)
    push!(cones_new, PSDTriangleConeT(cdim))
  end

end


function add_subblock_map!(H_I::Vector{Int}, clique_vertices::Array{Int}, row_start::Int)

  v = clique_vertices

  for j in eachindex(v), i in 1:j
      row = coord_to_upper_triangular_index((v[i], v[j]))
      push!(H_I, row_start + row - 1)
  end
end


function find_H_col_dimension(chordal_info::ChordalInfo{T}) where {T}

  cols, _ = get_decomposed_dim_and_overlaps(chordal_info)
  return cols

end 


