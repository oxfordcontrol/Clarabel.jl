# -----------------------------------------
# Functions related to clique tree based transformation (compact decomposition)
# see: Kim: Exploiting sparsity in linear and nonlinear matrix inequalities via 
# positive semidefinite matrix completion (2011), p.53
# -----------------------------------------

function augment_compact!(
    chordal_info::ChordalInfo,
    P::SparseMatrixCSC{T},
    q::Vector{T},
    A::SparseMatrixCSC{T},
    b::Vector{T},
    cones::Vector{SupportedCone}
  ) where {T}
    
    A_new, b_new, cones_new = find_compact_A_b_and_cones(chordal_info, A, b, cones)

    # how many variables did we add?
    nadd = A_new.n - A.n

    P_new = blockdiag(P, spzeros(nadd, nadd))
    q_new = vec([q; zeros(nadd)])

    return P_new, q_new, A_new, b_new, cones_new

end 

function find_compact_A_b_and_cones(
  chordal_info::ChordalInfo{T},
  A::SparseMatrixCSC{T},
  b::Vector{T},
  cones::Vector{SupportedCone}
) where{T}
spatterns = chordal_info.spatterns
  
  # determine number of final augmented matrix and number of overlapping entries
  Aa_m, Aa_n, n_overlaps = find_A_dimension(chordal_info, A)

  # allocate sparse components for the augmented A
  Aa_nnz = nnz(A) + 2 * n_overlaps
  Aa_I   = zeros(Int, Aa_nnz)
  Aa_J   = extra_columns(Aa_nnz, nnz(A) + 1, A.n + 1)
  Aa_V   = alternating_sequence(Aa_nnz, nnz(A) + 1)
  findnz!(Aa_I, Aa_J, Aa_V, A)

  # allocate sparse components for the augmented b
  # PJG: don't understand point of this sparse vector bs
  bs  = sparse(b)
  ba_I = zeros(Int, length(bs.nzval))
  ba_V = bs.nzval

  # preallocate the decomposed cones 
  cones_new = sizehint!(SupportedCone[], post_cone_count(chordal_info))

  # an iterator for the patterns.  We will expand cones assuming that 
  # they are non-decomposed until we reach an index that agrees with 
  # the internally stored coneidx of the next pattern.   
  patterns_iter = Iterators.Stateful(chordal_info.spatterns)
  row_ranges    = _make_rng_conesT(cones)

  row_ptr     = 1           # index to start of next cone in A_I
  overlap_ptr = nnz(A) + 1  # index to next row for +1, -1 overlap entries

  for (coneidx, (cone, row_range)) in enumerate(zip(cones,row_ranges))

    if !isempty(patterns_iter) && peek(patterns_iter).coneidx == coneidx

      @assert(isa(cone, PSDTriangleConeT))
      row_ptr, overlap_ptr = add_entries_with_sparsity_pattern!(
        Aa_I, ba_I, cones_new, A, bs, row_range, first(patterns_iter), row_ptr, overlap_ptr)

    else 
      row_ptr, overlap_ptr = add_entries_with_cone!(
        Aa_I, ba_I, cones_new, A, bs, row_range, cone, row_ptr, overlap_ptr)
    end 

  end

  A_new = allocate_sparse_matrix(Aa_I, Aa_J, Aa_V, Aa_m, Aa_n)
  b_new = Vector(SparseArrays._sparsevector!(ba_I, ba_V, Aa_m))

  A_new, b_new, cones_new
end 


"Given the row, column, and nzval vectors and dimensions, assemble the sparse matrix `Aa` of the decomposed problem in a slightly more memory efficient way."
function allocate_sparse_matrix(Aa_I::Array{Int, 1}, Aa_J::Array{Int, 1}, Aa_V::Array{Float64, 1}, mA::Int, nA::Int)
  csrrowptr = zeros(Int, mA + 1)
  csrcolval = zeros(Int, length(Aa_I))
  csrnzval = zeros(Float64, length(Aa_I))
  klasttouch = zeros(Int, nA + 1)
  csccolptr = zeros(Int, nA + 1)
  # sort_col_wise!(Aa_I, Aa_V, A.colptr, size(A, 2))
  #Aa = SparseMatrixCSC{Float64, Int}(mA, nA, Aa_J, Aa_I, Aa_V)
  Aa = SparseArrays.sparse!(Aa_I, Aa_J, Aa_V, mA, nA, +, klasttouch, csrrowptr, csrcolval, csrnzval, csccolptr, Aa_I, Aa_V )
end


# Handles all cones that are not decomposed by a sparsity pattern
function add_entries_with_cone!(
  Aa_I::Vector{Int}, 
  ba_I::Vector{Int}, 
  cones_new::Vector{SupportedCone}, 
  A::SparseMatrixCSC{T}, 
  b::SparseVector{T}, 
  row_range::UnitRange{Int}, 
  cone::SupportedCone,
  row_ptr::Int, overlap_ptr::Int
) where {T}

  n = A.n

  # populate b 
  offset = row_ptr - row_range.start

  row_range_col = get_rows(b, row_range)
  if row_range_col != 0:0
    for k in row_range_col
      ba_I[k] = b0.nzind[k] + offset
    end
  end

  #PJG: the cone map was previously populate here as well, but that 
  #is because the values where baked into the individual cones, which 
  #are the same ones that are used internally for projection.   Better 
  #to build up this mapping of decomposed cones to the original cones
  #in the ChordalInfo.   Note that that means that the cone_map object 
  #will have a number of records equal to the number of decomposed cones,
  #but we won't actually store the decomposed cones themselves in the 
  #chordal info.   It might also be possible to just work out this 
  #indexing on the fly where needed.

  # populate A 
  for col = 1:n
    # indices that store the rows in column col in A
    row_range_col = get_rows(A, col, row_range)
    if row_range_col != 0:0
      for k in row_range_col
        Aa_I[k] = A.rowval[k] + offset
      end
    end
  end

  push!(cones_new, deepcopy(cone))

  return row_ptr + nvars(cone), overlap_ptr
end



# Handle decomposable cones with a SparsityPattern. The row vectors A_I and b_I have to be edited in such a way 
# that entries of one clique appear contiguously.
function add_entries_with_sparsity_pattern!(
  A_I::Vector{Int}, 
  b_I::Vector{Int}, 
  cones_new::Vector{SupportedCone}, 
  A::SparseMatrixCSC{T}, 
  b::SparseVector{T}, 
  row_range::UnitRange{Int}, 
  spattern::SparsityPattern,
  row_ptr::Int, overlap_ptr::Int
) where {T}

  sntree   = spattern.sntree 
  ordering = spattern.ordering # LDL reordering 

  (_, n) = size(A) 

  # determine the row ranges for each of the subblocks
  clique_to_rows = clique_rows_map(row_ptr, sntree)

  # loop over cliques in descending topological order
  for i in num_cliques(sntree):-1:1

    # get supernodes and separators and undo the reordering
    # NB: these are now Vector, not VertexSet
    separator = sort!([spattern.ordering[v] for v in get_separator(sntree, i)])
    snode     = sort!([spattern.ordering[v] for v in get_snode(sntree, i)])

    # compute sorted block indices (i, j, flag) for this clique with an 
    # information flag whether an entry (i, j) is an overlap
    block_indices = get_block_indices(snode, separator, length(ordering))

    # If we encounter an overlap with a parent clique we have to be able to find the 
    # location of the overlapping entry. Therefore load and reorder the parent clique
    if i == num_cliques(sntree)
      parent_rows   = 0:0
      parent_clique = Int[]
    else
      parent_index  = get_clique_parent(sntree, i)
      parent_rows   = clique_to_rows[parent_index]
      parent_clique = ([spattern.ordering[v] for v in get_clique_by_index(sntree, parent_index)])
      sort!(parent_clique)
    end

    # Loop over all the columns and shift the rows in A_I and b_I according to the clique structure
    for col = 1:n
      row_range_col = get_rows(A, col, row_range)
      row_range_b = col == 1 ? get_rows(b, row_range) : 0:0

      overlap_ptr = add_clique_entries!(
        A_I, b_I, A.rowval, b.nzind, block_indices, 
        parent_clique, parent_rows, col, 
        row_ptr, overlap_ptr, row_range, row_range_col, row_range_b)

    end

    # create and add new cone for subblock
    cone_dim = get_nblk(sntree, i)
    num_rows = triangular_number(cone_dim)

    #PJG: new PSD cones are created here. 
    #In COSMO, they are tagged with their tree and clique numbers
    push!(cones_new, PSDTriangleConeT(cone_dim))
    row_ptr += num_rows
    
  end

  return return row_ptr, overlap_ptr

end



" Loop over all entries (i, j) in the clique and either set the correct row in `A_I` and `b_I` if (i, j) is not an overlap,
 or add an overlap column with (-1 and +1) in the correct positions."
function add_clique_entries!(
  A_I::Vector{Int}, 
  b_I::Vector{Int}, 
  A_rowval::Vector{Int}, 
  b_nzind::Vector{Int}, 
  block_indices::Vector{Tuple{Int, Int, Bool}},  
  parent_clique::Vector{Int}, 
  parent_rows::UnitRange{Int}, 
  col::Int,  
  row_ptr::Int, 
  overlap_ptr::Int, 
  row_range::UnitRange{Int}, 
  row_range_col::UnitRange{Int}, 
  row_range_b::UnitRange{Int}
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


" Given the nominal entry position `k = linearindex(i, j)` find and modify with `new_row_val` 
  the actual location of that entry in the global row vector `rowval`."
function modify_clique_rows!(
  v::Vector{Int}, 
  k::Int, 
  rowval::Vector{Int}, 
  new_row_val::Int, 
  row_range::UnitRange{Int}, 
  row_range_col::UnitRange{Int}
)

  row_0 = get_row_index(k, rowval, row_range, row_range_col)
  # row_0 happens when (i, j) references an edge that was added by merging cliques, 
  # the corresponding value will be zero and can be disregarded
  if row_0 != 0
    v[row_0] = new_row_val
  end
  return nothing
end


"Given the svec index `k` and an offset `row_range_col.start`, return the location of the (i, j)th entry in the row vector `rowval`."
function get_row_index(
  k::Int, 
  rowval::Vector{Int}, 
  row_range::UnitRange{Int}, 
  row_range_col::UnitRange{Int}

  )
  row_range_col == 0:0 && return 0
  k_shift = row_range.start + k - 1

  # determine upper set boundary of where the row could be
  u = min(row_range_col.stop, row_range_col.start + k_shift - 1)

  # find index of first entry >= k, starting in the interval [l, u]
  # if no, entry is >= k, returns u + 1
  r = searchsortedfirst(rowval, k_shift, row_range_col.start, u, Base.Order.Forward)
  # if no r s.t. rowval[r] = k_shift was found that means that the (i, j)th entry represents an edded edge (zero) from clique merging
  if r > u || rowval[r] != k_shift
    return 0
  else
    return r
  end
end


" Find the index of k=svec(i, j) in the parent clique `par_clique`."
function parent_block_indices(parent_clique::Vector{Int}, i::Int, j::Int)
  ir = searchsortedfirst(parent_clique, i)
  jr = searchsortedfirst(parent_clique, j)
  return coord_to_upper_triangular_index((ir, jr))
end


"""
    (snd::Array{Int}, sep::Array{Int})

For a clique consisting of supernodes `snd` and seperators `sep`, compute all the indices (i, j) of the corresponding matrix block
in the format (i, j, flag) where flag is equal to false if entry (i, j) corresponds to an overlap of the clique and true otherwise.

`Nv` is the number of vertices in the graph that we are trying to decompose.
"""
function get_block_indices(snode::Array{Int}, separator::Array{Int}, nv::Int)
  
  N = length(separator) + length(snode)

  block_indices = sizehint!(Tuple{Int, Int, Bool}[],triangular_number(N))
  ind = 1

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



"Return the row ranges of each clique after the decomposition of `C` shifted by `row_start`."
function clique_rows_map(row_start::Int, sntree::SuperNodeTree)
  
  n_cliques = num_cliques(sntree)
  rows = sizehint!(UnitRange{Int}[],  n_cliques)
  inds = sizehint!(Int[], n_cliques)

  for i in n_cliques:-1:1
    num_rows = triangular_number(get_nblk(sntree, i))
    push!(rows, row_start:row_start+num_rows-1)
    push!(inds, sntree.snode_post[i])
    row_start += num_rows
  end

  return Dict(inds .=> rows)
end


function get_rows(b::SparseVector, row_range::UnitRange{Int})
    rows = b.nzind
    if length(rows) > 0
  
      s = searchsortedfirst(rows, row_range.start)
      if s == length(rows) + 1
        return 0:0
      end
  
      if rows[s] > row_range.stop || s == 0
          return 0:0
      else
        e = searchsortedlast(rows, row_range.stop)
        return s:e
      end
    else
      return 0:0
    end
  
  end
  
  function get_rows(A::SparseMatrixCSC, col::Int, row_range::UnitRange{Int})
    colrange = A.colptr[col]:(A.colptr[col + 1]-1)
  
    # if the column has entries
    if colrange.start <= colrange.stop
      # create a view into the row values of column col
      rows = view(A.rowval, colrange)
      # find the rows within row_start:row_start+C.dim-1
      # s: index of first entry in rows >= row_start
      s = searchsortedfirst(rows, row_range.start)
  
      # if no entry in that row_range
      if s == length(rows) + 1
        return 0:0
      end
      if rows[s] > row_range.stop || s == 0
        return 0:0
      else
        # e: index of last value in rows <= row_start + C.dim - 1
        e = searchsortedlast(rows, row_range.stop)
        return colrange[s]:colrange[e]
      end
    else
      return 0:0
    end
  end


  "Returns the appropriate amount of memory for `A.nzval`, including, starting from `n_start`, the (+1 -1) entries for the overlaps."
function alternating_sequence(total_length::Int, n_start::Int)
  v = ones(Float64, total_length)
  for i= n_start + 1:2:length(v)
    v[i] = -1
  end
  return v
end


"Returns the appropriate amount of memory for the columns of the augmented problem matrix `A`, including, starting from `n_start`, the columns for the (+1 -1) entries for the overlaps."
function extra_columns(total_length::Int, n_start::Int, start_val::Int)
  v = zeros(Int, total_length)
  for i = n_start:2:length(v)-1
    v[i]   = start_val
    v[i+1] = start_val
    start_val += 1
  end
  return v
end


"Given a sparse matrix `S`, write the columns and non-zero values into the first `numnz` entries of `J` and `V`."
function findnz!(I::Vector{Ti}, J::Vector{Ti}, V::Vector{Tv}, S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    numnz = nnz(S)
    count = 1
    @inbounds for col = 1 : S.n, k = S.colptr[col] : (S.colptr[col+1]-1)
       # I[count] = S.rowval[k]
        J[count] = col
        V[count] = S.nzval[k]
        count += 1
    end
end


function find_A_dimension(chordal_info::ChordalInfo{T}, A::SparseMatrixCSC{T}) where {T}


  dim, num_overlaps  = get_decomposed_dim_and_overlaps(chordal_info)

  rows = dim 
  cols = A.n + num_overlaps

  return rows, cols, num_overlaps

end 