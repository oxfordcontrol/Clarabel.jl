# -----------------------------------------
# Functions related to clique tree based transformation (compact decomposition)
# see: Kim: Exploiting sparsity in linear and nonlinear matrix inequalities via 
# positive semidefinite matrix completion (2011), p.53
# -----------------------------------------

const BlockOverlapTriplet = Tuple{DefaultInt, DefaultInt, Bool}

function decomp_augment_compact!(
    chordal_info::ChordalInfo,
    P::SparseMatrixCSC{T},
    q::Vector{T},
    A::SparseMatrixCSC{T},
    b::Vector{T},
  ) where {T}
    
    A_new, b_new, cones_new = find_compact_A_b_and_cones(chordal_info, A, b)

    # how many variables did we add?
    nadd = A_new.n - A.n

    P_new = blockdiag(P, spzeros(T, nadd, nadd))

    q_new = zeros(T, length(q) + nadd)
    q_new[1:length(q)] .= q

    return P_new, q_new, A_new, b_new, cones_new

end 

function find_compact_A_b_and_cones(
  chordal_info::ChordalInfo{T},
  A::SparseMatrixCSC{T},
  b::Vector{T},
) where{T}

  # the cones that we used to form the decomposition 
  cones = chordal_info.init_cones

  # determine number of final augmented matrix and number of overlapping entries
  Aa_m, Aa_n, n_overlaps = find_A_dimension(chordal_info, A)

  # allocate sparse components for the augmented A
  Aa_nnz = nnz(A) + 2 * n_overlaps
  Aa_I   = zeros(DefaultInt, Aa_nnz)
  Aa_J   = extra_columns(Aa_nnz, nnz(A) + 1, A.n + 1)
  Aa_V   = alternating_sequence(T, Aa_nnz, nnz(A) + 1)

  findnz!(Aa_J, Aa_V, A)

  # allocate sparse components for the augmented b
  bs   = sparse(b)
  ba_I = zeros(DefaultInt, length(bs.nzval))
  ba_V = bs.nzval

  # preallocate the decomposed cones and the mapping 
  # from decomposed cones back to the originals
  n_decomposed = final_cone_count(chordal_info)
  cones_new = sizehint!(SupportedCone[], n_decomposed)
  cone_maps = sizehint!(ConeMapEntry[],  n_decomposed)

  # an enumerate-like mutable iterator for the patterns.  We will expand cones 
  # assuming that they are non-decomposed until we reach an index that agrees  
  # the internally stored orig_index of the next pattern.   

  patterns_iter  = Iterators.Stateful(chordal_info.spatterns)
  patterns_count = Iterators.Stateful(eachindex(chordal_info.spatterns))
  row_ranges     = rng_cones_iterator(cones)

  row_ptr     = 1           # index to start of next cone in A_I
  overlap_ptr = nnz(A) + 1  # index to next row for +1, -1 overlap entries

  for (coneidx, (cone, row_range)) in enumerate(zip(cones,row_ranges))

    if !isempty(patterns_iter) && peek(patterns_iter).orig_index == coneidx

      @assert(isa(cone, PSDTriangleConeT))
      row_ptr, overlap_ptr = add_entries_with_sparsity_pattern!(
        Aa_I, ba_I, cones_new, cone_maps, A, bs, row_range, 
        first(patterns_iter), first(patterns_count), 
        row_ptr, overlap_ptr)
    else 
      row_ptr, overlap_ptr = add_entries_with_cone!(
        Aa_I, ba_I, cones_new, cone_maps, A, bs, row_range, cone, row_ptr, overlap_ptr)
    end 

  end

  # save the cone_maps for use when reconstructing 
  # solution to the original problem
  chordal_info.cone_maps = cone_maps

  A_new = allocate_sparse_matrix(Aa_I, Aa_J, Aa_V, Aa_m, Aa_n)
  b_new = Vector(SparseArrays._sparsevector!(ba_I, ba_V, Aa_m))

  A_new, b_new, cones_new
end 

# find the dimension of the `compact' form `A' matrix and its number of overlaps 
function find_A_dimension(chordal_info::ChordalInfo{T}, A::SparseMatrixCSC{T}) where {T}

  dim, num_overlaps  = get_decomposed_dim_and_overlaps(chordal_info)

  rows = dim 
  cols = A.n + num_overlaps

  return rows, cols, num_overlaps

end 


# Given the row, column, and nzval vectors and dimensions, assembles the sparse matrix `Aa` 
# of the decomposed problem in a slightly more memory efficient way.
function allocate_sparse_matrix(
  Aa_I::Vector{DefaultInt}, 
  Aa_J::Vector{DefaultInt}, 
  Aa_V::Vector{T}, 
  mA::DefaultInt, nA::DefaultInt
) where {T}
  csrrowptr = zeros(DefaultInt, mA + 1)
  csrcolval = zeros(DefaultInt, length(Aa_I))
  csrnzval = zeros(T, length(Aa_I))
  klasttouch = zeros(DefaultInt, nA + 1)
  csccolptr = zeros(DefaultInt, nA + 1)

  SparseArrays.sparse!(Aa_I, Aa_J, Aa_V, mA, nA, +, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr, Aa_I, Aa_V )

end


# Handles all cones that are not decomposed by a sparsity pattern
function add_entries_with_cone!(
  Aa_I::Vector{DefaultInt}, 
  ba_I::Vector{DefaultInt}, 
  cones_new::Vector{SupportedCone}, 
  cone_maps::Vector{ConeMapEntry},
  A::SparseMatrixCSC{T}, 
  b::SparseVector{T}, 
  row_range::UnitRange{DefaultInt}, 
  cone::SupportedCone,
  row_ptr::DefaultInt, overlap_ptr::DefaultInt
) where {T}

  n = A.n
  offset = row_ptr - row_range.start


  # populate b 
  row_range_col = get_rows_vec(b, row_range)
  if !isnothing(row_range_col)
    for k in row_range_col
      ba_I[k] = b.nzind[k] + offset
    end
  end

  # populate A 
  for col = 1:n
    # indices that store the rows in column col in A
    row_range_col = get_rows_mat(A, col, row_range)
    if !isnothing(row_range_col)
      for k in row_range_col
        Aa_I[k] = A.rowval[k] + offset
      end
    end
  end

  # here we make a copy of the cone
  push!(cones_new, deepcopy(cone))

  # since this cone is standalone and not decomposed, the index 
  # of its origin cone must be either one more than the previous one,
  # or 1 (zero in rust) if it's the first 

  orig_index = isempty(cone_maps) ? 1 : cone_maps[end].orig_index + 1
  push!(cone_maps, ConeMapEntry(orig_index, nothing))

  return row_ptr + nvars(cone), overlap_ptr

end



# Handle decomposable cones with a SparsityPattern. The row vectors A_I and b_I 
# have to be edited in such a way that entries of one clique appear contiguously.
function add_entries_with_sparsity_pattern!(
  A_I::Vector{DefaultInt}, 
  b_I::Vector{DefaultInt}, 
  cones_new::Vector{SupportedCone}, 
  cone_maps::Vector{ConeMapEntry},
  A::SparseMatrixCSC{T}, 
  b::SparseVector{T}, 
  row_range::UnitRange{DefaultInt}, 
  spattern::SparsityPattern,
  spattern_index::DefaultInt,
  row_ptr::DefaultInt, overlap_ptr::DefaultInt
) where {T}

  sntree   = spattern.sntree 
  ordering = spattern.ordering  

  (_, n) = size(A) 

  # determine the row ranges for each of the subblocks
  clique_to_rows = clique_rows_map(row_ptr, sntree)

  # loop over cliques in descending topological order
  for i in (sntree.n_cliques):-1:1

    # get supernodes and separators and undo the reordering
    # NB: these are now Vector, not VertexSet
    separator = sort!([spattern.ordering[v] for v in get_separators(sntree, i)])
    snode     = sort!([spattern.ordering[v] for v in get_snode(sntree, i)])

    # compute sorted block indices (i, j, flag) for this clique with an 
    # information flag whether an entry (i, j) is an overlap
    block_indices = get_block_indices(snode, separator, length(ordering))

    # If we encounter an overlap with a parent clique we have to be able to find the 
    # location of the overlapping entry. Therefore load and reorder the parent clique
    if i == sntree.n_cliques
      parent_rows   = 0:0
      parent_clique = DefaultInt[]
    else
      parent_index  = get_clique_parent(sntree, i)
      parent_rows   = clique_to_rows[parent_index]
      parent_clique = [spattern.ordering[v] for v in get_clique_by_index(sntree, parent_index)]
      sort!(parent_clique)
    end

    # Loop over all the columns and shift the rows in A_I and b_I according to the clique structure
    for col = 1:n

        row_range_col = get_rows_mat(A, col, row_range)
        row_range_b = col == 1 ? get_rows_vec(b, row_range) : 0:0

        #PJG this method for producing empty ranges is gross
        #should just be Nothing
        row_range_col = isnothing(row_range_col) ? (1:0) : row_range_col
        row_range_b = isnothing(row_range_b) ? (1:0) : row_range_b

        overlap_ptr = add_clique_entries!(
        A_I, b_I, A.rowval, b.nzind, block_indices, 
        parent_clique, parent_rows, col, 
        row_ptr, overlap_ptr, row_range, row_range_col, row_range_b)

    end

    # create new PSD cones for the subblocks, and tag them 
    # with their tree and clique number

    cone_dim = get_nblk(sntree, i)
    push!(cones_new, PSDTriangleConeT(cone_dim))
    push!(cone_maps, ConeMapEntry(spattern.orig_index, (spattern_index, i)))

    row_ptr += triangular_number(cone_dim)
    
  end

  return row_ptr, overlap_ptr

end



# Loop over all entries (i, j) in the clique and either set the correct row in `A_I` and `b_I` 
# if (i, j) is not an overlap,or add an overlap column with (-1 and +1) in the correct positions.

function add_clique_entries!(
  A_I::Vector{DefaultInt}, 
  b_I::Vector{DefaultInt}, 
  A_rowval::Vector{DefaultInt}, 
  b_nzind::Vector{DefaultInt}, 
  block_indices::Vector{BlockOverlapTriplet},  
  parent_clique::Vector{DefaultInt}, 
  parent_rows::UnitRange{DefaultInt}, 
  col::DefaultInt,  
  row_ptr::DefaultInt, 
  overlap_ptr::DefaultInt, 
  row_range::UnitRange{DefaultInt}, 
  row_range_col::UnitRange{DefaultInt}, 
  row_range_b::UnitRange{DefaultInt}
)

  counter = 0
  for block_idx in block_indices
    new_row_val = row_ptr + counter
    (i,j,is_overlap) = block_idx
    # a block index that corresponds to an overlap
    if is_overlap
      if col == 1
        # this creates the +1 entry
        A_I[overlap_ptr] = new_row_val 
        # this creates the -1 entry
        A_I[overlap_ptr + 1] = parent_rows.start + parent_block_indices(parent_clique, i, j) - 1 
        overlap_ptr += 2
      end
    else
      k = coord_to_upper_triangular_index((i, j))
      modify_clique_rows!(A_I, k, A_rowval,  new_row_val, row_range, row_range_col)
      col == 1 && modify_clique_rows!(b_I, k, b_nzind, new_row_val, row_range, row_range_b)

    end
    counter += 1

  end
  return overlap_ptr
end


# Given the nominal entry position `k = linearindex(i, j)` find and modify with `new_row_val` 
# the actual location of that entry in the global row vector `rowval`.

function modify_clique_rows!(
  v::Vector{DefaultInt}, 
  k::DefaultInt, 
  rowval::Vector{DefaultInt}, 
  new_row_val::DefaultInt, 
  row_range::UnitRange{DefaultInt}, 
  row_range_col::UnitRange{DefaultInt}
)

  row_0 = get_row_index(k, rowval, row_range, row_range_col)
  # row_0 happens when (i, j) references an edge that was added by merging cliques, 
  # the corresponding value will be zero and can be disregarded
  if row_0 != 0
    v[row_0] = new_row_val
  end
  return nothing
end


# Given the svec index `k` and an offset `row_range_col.start`, return the location of the 
# (i, j)th entry in the row vector `rowval`.
function get_row_index(
  k::DefaultInt, 
  rowval::Vector{DefaultInt}, 
  row_range::UnitRange{DefaultInt}, 
  row_range_col::UnitRange{DefaultInt}
  )

  #PJG: possible logic error here, since we don't pass down a Union type 
  #this far.  Fix this and use Option everywhere, or short circuit 
  #higher up the call stack.   Should also return Nothing instead of 0
  isnothing(row_range_col) && return 0

  k_shift = row_range.start + k - 1

  # determine upper set boundary of where the row could be
  u = min(row_range_col.stop, row_range_col.start + k_shift - 1)

  # find index of first entry >= k, starting in the interval [l, u]
  # if no, entry is >= k, returns u + 1
  r = searchsortedfirst(rowval, k_shift, row_range_col.start, u, Base.Order.Forward)

  # if no r s.t. rowval[r] = k_shift was found, that means that the 
  # (i, j)th entry represents an edded edge (zero) from clique merging
  if r > u || rowval[r] != k_shift
    return 0
  else
    return r
  end
   

end



# Find the index of k=svec(i, j) in the parent clique `par_clique`.#

function parent_block_indices(parent_clique::Vector{DefaultInt}, i::DefaultInt, j::DefaultInt)
  ir = searchsortedfirst(parent_clique, i)
  jr = searchsortedfirst(parent_clique, j)
  return coord_to_upper_triangular_index((ir, jr))
end



# Given a cliques supernodes and separators, compute all the indices (i, j) of the corresponding matrix block
# in the format (i, j, flag), where flag is equal to false if entry (i, j) corresponds to an overlap of the 
# clique and true otherwise.

# `nv` is the number of vertices in the graph that we are trying to decompose.

function get_block_indices(snode::Array{DefaultInt}, separator::Array{DefaultInt}, nv::DefaultInt)
  
  N = length(separator) + length(snode)

  block_indices = sizehint!(BlockOverlapTriplet[],triangular_number(N))

  for j in separator, i in separator
    if i <= j
      push!(block_indices,(i, j, true))
    end
  end

  for j in snode, i in snode
    if i <= j
      push!(block_indices,(i, j, false))
    end
  end

  for i in snode
    for j in separator
      push!(block_indices,(min(i, j), max(i, j), false))
    end
  end

  sort!(block_indices, by = x -> x[2] * nv + x[1] )
  return block_indices
end



# Return the row ranges of each clique after the decomposition, shifted by `row_start`.

function clique_rows_map(row_start::DefaultInt, sntree::SuperNodeTree)
  
  n_cliques = sntree.n_cliques

  # PJG: rows / inds not necessary here.   Should just size hint 
  # on the output Dict and push the entries directly to it
  rows = sizehint!(UnitRange{DefaultInt}[],  n_cliques)
  inds = sizehint!(DefaultInt[], n_cliques)

  for i in n_cliques:-1:1
    num_rows = triangular_number(get_nblk(sntree, i))
    push!(rows, row_start:row_start+num_rows-1)
    push!(inds, sntree.snode_post[i])
    row_start += num_rows
  end

  return Dict(inds .=> rows)
end

function get_rows_subset(rows, row_range)

  if length(rows) == 0
    return nothing
  end

  s = searchsortedfirst(rows, row_range.start)
  if s == length(rows) + 1
    return nothing
  end

  if rows[s] > row_range.stop || s == 0
      return nothing
  else
    e = searchsortedlast(rows, row_range.stop)
    return s:e
  end

end

function get_rows_vec(b::SparseVector, row_range::UnitRange{DefaultInt})
  
    get_rows_subset(b.nzind, row_range)
  
  end
  
  function get_rows_mat(A::SparseMatrixCSC, col::DefaultInt, row_range::UnitRange{DefaultInt})

    colrange = A.colptr[col]:(A.colptr[col + 1]-1)
    rows = view(A.rowval, colrange)
    se = get_rows_subset(rows, row_range)

    if isnothing(se)
      return nothing
    end

    colrange[se.start]:colrange[se.stop]

  end


# Returns the appropriate amount of memory for `A.nzval`, including, starting from `n_start`, 
# the (+1 -1) entries for the overlaps.

function alternating_sequence(T, total_length::DefaultInt, n_start::DefaultInt)
  v = ones(T, total_length)
  for i= (n_start + 1):2:length(v)
    v[i] = -one(T)
  end
  return v
end


# Returns the appropriate amount of memory for the columns of the augmented problem matrix `A`, 
# including, starting from `n_start`, the columns for the (+1 -1) entries for the overlaps.

function extra_columns(total_length::DefaultInt, n_start::DefaultInt, start_val::DefaultInt)
  v = zeros(DefaultInt, total_length)
  for i = n_start:2:length(v)-1
    v[i]   = start_val
    v[i+1] = start_val
    start_val += 1
  end
  return v
end


# Given sparse matrix components, write the columns and non-zero values into the first `numnz` entries of `J` and `V`.

function findnz!(J::Vector{Ti}, V::Vector{Tv}, S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    count = 1
    @inbounds for col = 1 : S.n, k = S.colptr[col] : (S.colptr[col+1]-1)
        J[count] = col
        V[count] = S.nzval[k]
        count += 1
    end
end


# Intentionally defined here separately from the other SuperNodeTree functions.
# This returns a clique directly from index i, rather than accessing the 
# snode and separators via the postordering.   Only used in the compact  
# problem decomposition functions 

function get_clique_by_index(sntree::SuperNodeTree, i::DefaultInt)
	return union(sntree.snode[i], sntree.separators[i])
end
