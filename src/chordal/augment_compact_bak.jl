# -----------------------------------------
# Functions related to clique tree based transformation (compact decomposition)
# see: Kim: Exploiting sparsity in linear and nonlinear matrix inequalities via 
# positive semidefinite matrix completion (2011), p.53
# -----------------------------------------


function augment_compact!(
    chordal_info::ChordalInfo,
    P::SparseMatrixCSC{T},
    q::Vector{T},
    A::SparsematrixCSC{T},
    b::Vector{T},
    cones::Vector{SupportedCone}
  ) where {T}

    spatterns = chordal_info.spatterns
  
    # determine number of final augmented matrix and number of overlapping entries
    mA, nA, num_overlapping_entries = find_A_dimension(ws.p.model_size[2], ws.p.C.sets, ws.ci.sp_arr)
    # find number of decomposed and total sets and allocate structure for new compositve convex set
    num_total, num_new_psd_cones = COSMO.num_cone_decomposition(ws)
  
    C_new = Array{COSMO.AbstractConvexSet{T}}(undef, num_total - 1)
  
    # allocate memory for Aa_I, Aa_J, Aa_V, b_I
    nz = nnz(A)
    Aa_I = zeros(Int, nz + 2 * num_overlapping_entries)
    Aa_J = extra_columns(nz + 2 * num_overlapping_entries, nz + 1, ws.p.model_size[2] + 1)
    Aa_V =  alternating_sequence(nz + 2 * num_overlapping_entries, nz + 1)
    findnz!(Aa_I, Aa_J, Aa_V, A)
    bs = sparse(b)
    b_I = zeros(Int, length(bs.nzval))
    b_V = bs.nzval
  
  
    row_ranges = COSMO.get_set_indices(cones) # the row ranges of each cone in the original problem
    row_ptr = 1 # a continuously updated pointer to the start of the first entry of a cone in A_I
    overlap_ptr = nz + 1 # a continuously updated pointer to the next free entry to enter the rows for the +1, -1 overlap entries
  
    sp_ind = 1 # every decomposable cone is linked to a sparsity_pattern with index sp_ind
    set_ind = 1 # the set index counter set_ind is used to create a map of the set number in the original problem to the set number in the decomposed problem
  
    # Main Loop: Loop over the cones of the original problem and add entries to the row vectors Aa_I and b_I
    for (k, C) in enumerate(cones)
      row_range = row_ranges[k]
      row_ptr, overlap_ptr, set_ind, sp_ind = COSMO.add_entries!(Aa_I, b_I, C_new, row_ptr, A, bs, row_range, overlap_ptr, set_ind, sp_ind, sp_arr, C, k, ws.ci.cone_map)
    end
  
    ws.p.A = allocate_sparse_matrix(Aa_I, Aa_J, Aa_V, mA, nA)
    ws.p.b = Vector(SparseArrays._sparsevector!(b_I, b_V, mA))
    ws.p.P = blockdiag(P, spzeros(num_overlapping_entries, num_overlapping_entries))
    ws.p.q = vec([q; zeros(num_overlapping_entries)])
    ws.p.model_size[1] = size(ws.p.A, 1)
    ws.p.model_size[2] = size(ws.p.A, 2)
    ws.p.C = COSMO.CompositeConvexSet(C_new)
    return nothing

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


"Return the dimension of the problem after a clique tree based decomposition, given the sparsity patterns in `sp_arr`."
function find_A_dimension(n_original::Int, sets, sp_arr)
  num_cols = n_original
  num_overlapping_entries = 0
  num_rows = 0
  sp_arr_ind = 1

    #PJG: here C is passed so that a multidispatch mechanism 
    #can catch non-decomposable cones (including dense PSD),
    #and doesn't increment the sp_arr_index counter.  Also 
    #passed in to hel differentiate full from triangular cones.  
    #I have simplified this a lot in the "H" case for the 
    #traditional decomp.   Check implementation there for hints

  for C in sets
    dim, overlaps, sp_arr_ind = decomposed_dim(C, sp_arr, sp_arr_ind)
    num_rows += dim
    num_overlapping_entries += overlaps
  end
  return num_rows::Int, (num_cols + num_overlapping_entries)::Int, num_overlapping_entries
end



# This method handles decomposable cones, e.g. PsdConeTriangle. The row vectors A_I and b_I have to be edited in such a way that entries of one clique are appear continously
function add_entries!(A_I::Array{Int, 1}, b_I::Array{Int, 1}, C_new::Array{COSMO.AbstractConvexSet{Float64}, 1}, row_ptr::Int, A0::SparseMatrixCSC, b0::SparseVector, row_range::UnitRange{Int}, overlap_ptr::Int, set_ind::Int, sp_ind::Int,
    sp_arr::Array{SparsityPattern, 1},  C::PsdConeTriangle{Float64}, k::Int,  cone_map::Dict{Int, Int})
  
    sp = sp_arr[sp_ind] # The SparsityPattern correspondig to this cone C
    sntree = sp.sntree # The Supernodal Elimination Tree that stores information about the cliques
    ordering = sp.ordering # A reordering that was applied by the LDL routine
    N_v = length(ordering)
    m, n = size(A0) # Dimensions of the original problem
  
    # determine the row ranges for each of the subblocks
    clique_to_rows = COSMO.clique_rows_map(row_ptr, sntree, C)
  
    # loop over cliques in descending topological order
    for iii = num_cliques(sntree):-1:1
  
      # get supernodes and separators and undo the reordering
      sep = map(v -> ordering[v], get_sep(sntree, iii))
      isa(sep, Array{Any, 1}) && (sep = Int[])
      snd = map(v -> ordering[v], get_snd(sntree, iii))
      # compute sorted block indices (i, j, flag) for this clique with an information flag whether an entry (i, j) is an overlap
      block_indices = COSMO.get_block_indices(snd, sep, N_v)
  
      # If we encounter an overlap with a parent clique we have to be able to find the location of the overlapping entry
      # Therefore load and reorder the parent clique
      if iii == num_cliques(sntree)
        par_clique = Int[]
        par_rows = 0:0
      else
        par_ind = COSMO.get_clique_par(sntree, iii)
        par_rows = clique_to_rows[par_ind]

        # PJG: this should produce a vector that can 
        # be sorted on the next like 
        par_clique = map(v -> ordering[v], get_clique_by_ind(sntree, par_ind))
  
        #PJG: here it matters I think that the parent clique is an array, 
        #and so therefore can't be an unordered set.   That means that 
        #the call to get_clique_by_ind, which computes a union of 
        #snode and separators, must itself be operating on vectors 
        sort!(par_clique)
      end
  
      # Loop over all the columns and shift the rows in A_I and b_I according to the clique strucutre
      for col = 1:n
        row_range_col = COSMO.get_rows(A0, col, row_range)
        row_range_b = col == 1 ? COSMO.get_rows(b0, row_range) : 0:0
        overlap_ptr = add_clique_entries!(A_I, b_I, A0.rowval, b0.nzind, block_indices, par_clique, par_rows, col, C.sqrt_dim, row_ptr, overlap_ptr, row_range, row_range_col, row_range_b)
      end
  
      # PJG: new PSD cone objects are generated here for the 
      # decomposition, getting effectively a new vector of SupportedCone
      # this should probably be done during the constructor 
      # instead, since it seems unrelated to the above. 
  
      # create and add new cone for subblock
      num_rows = get_blk_rows(get_nBlk(sntree, iii), C)
      cone_map[set_ind] = k
  
      #PJG: new PSD cones are created here, and are tagged with their tree and clique numbers
      C_new[set_ind] = typeof(C)(num_rows, sp_ind, iii)
      row_ptr += num_rows
      set_ind += 1
    end
    return row_ptr, overlap_ptr, set_ind, sp_ind + 1
  end
  
  " Loop over all entries (i, j) in the clique and either set the correct row in `A_I` and `b_I` if (i, j) is not an overlap or add an overlap column with (-1 and +1) in the correct positions."
  function add_clique_entries!(A_I::Array{Int, 1}, b_I::Array{Int, 1}, A_rowval::Array{Int}, b_nzind::Array{Int, 1}, block_indices::Array{Tuple{Int, Int, Int},1},  par_clique::Array{Int, 1}, par_rows::UnitRange{Int}, col::Int,  C_sqrt_dim::Int, row_ptr::Int, overlap_ptr::Int, row_range::UnitRange{Int}, row_range_col::UnitRange{Int}, row_range_b::UnitRange{Int})
    counter = 0
    for block_idx in block_indices
      new_row_val = row_ptr + counter
      # a block index that corresponds to an overlap
      if block_idx[3] == 0
        if col == 1
          i = block_idx[1]
          j = block_idx[2]
          A_I[overlap_ptr] = new_row_val # this creates the +1 entry
          A_I[overlap_ptr + 1] = par_rows.start + COSMO.parent_block_indices(par_clique, i, j) - 1 # this creates the -1 entry
          overlap_ptr += 2
        end
      else
        # (i, j) of the clique
        i = block_idx[1]
        j = block_idx[2]
        # k = svec(i, j)
        k = COSMO.mat_to_svec_ind(i, j)
        modify_clique_rows!(A_I, k, A_rowval, C_sqrt_dim, new_row_val, row_range, row_range_col)
        col == 1 && modify_clique_rows!(b_I, k, b_nzind, C_sqrt_dim, new_row_val, row_range, row_range_b)
      end
    counter += 1
    end
    return overlap_ptr
  end


" Given the nominal entry position `k = svec(i, j)` find and modify with `new_row_val` the actual location of that entry in the global row vector `rowval`."
function modify_clique_rows!(v::Array{Int, 1}, k::Int, rowval::Array{Int, 1}, C_sqrt_dim::Int, new_row_val::Int, row_range::UnitRange{Int}, row_range_col::UnitRange{Int})
  row_0 = COSMO.get_row_index(k, rowval, C_sqrt_dim, row_range, row_range_col)
  # row_0 happens when (i, j) references an edge that was added by merging cliques, the corresponding value will be zero
  # and can be disregarded
  if row_0 != 0
    v[row_0] = new_row_val
  end
  return nothing
end


"Given the svec index `k` and an offset `row_range_col.start`, return the location of the (i, j)th entry in the row vector `rowval`."
function get_row_index(k::Int, rowval::Array{Int, 1}, sqrt_dim::Int, row_range::UnitRange{Int}, row_range_col::UnitRange{Int})
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
function parent_block_indices(par_clique::Array{Int, 1}, i::Int, j::Int)
  ir = searchsortedfirst(par_clique, i)
  jr = searchsortedfirst(par_clique, j)
  return COSMO.mat_to_svec_ind(ir, jr)
end



"""
    get_block_indices(snd::Array{Int}, sep::Array{Int})

For a clique consisting of supernodes `snd` and seperators `sep`, compute all the indices (i, j) of the corresponding matrix block
in the format (i, j, flag) where flag is equal to 0 if entry (i, j) corresponds to an overlap of the clique and 1 otherwise.

`Nv` is the number of vertices in the graph that we are trying to decompose.
"""
function get_block_indices(snd::Array{Int}, sep::Array{Int}, Nv::Int)
  N = length(sep) + length(snd)
  d = div(N * (N + 1), 2)

  block_indices = Array{Tuple{Int, Int, Int}, 1}(undef, d)
  ind = 1

  for j in sep, i in sep
    if i <= j
      block_indices[ind] = (i, j, 0)
      ind += 1
    end
  end

  for j in snd, i in snd
    if i <= j
      block_indices[ind] = (i, j, 1)
      ind += 1
    end
  end

  for i in snd
    for j in sep
      block_indices[ind] = (min(i, j), max(i, j), 1)
      ind += 1
    end
  end

  sort!(block_indices, by = x -> x[2] * Nv + x[1] )
  return block_indices
end



"Return the row ranges of each clique after the decomposition of `C` shifted by `row_start`."
function clique_rows_map(row_start::Int, sntree::SuperNodeTree, C::DecomposableCones{<:Real})
  Nc = num_cliques(sntree)
  rows = Array{UnitRange{Int}}(undef,  Nc)
  ind = zeros(Int, Nc)
  for iii = Nc:-1:1
    num_rows = COSMO.get_blk_rows(COSMO.get_nBlk(sntree, iii), C)
    rows[iii] = row_start:row_start+num_rows-1
    ind[iii] = sntree.snd_post[iii]
    row_start += num_rows
  end
  return Dict(ind .=> rows)
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
  for i=n_start + 1:2:length(v)
    v[i] = -1
  end
  return v
end


"Returns the appropriate amount of memory for the columns of the augmented problem matrix `A`, including, starting from `n_start`, the columns for the (+1 -1) entries for the overlaps."
function extra_columns(total_length::Int, n_start::Int, start_val::Int)
  v = zeros(Int, total_length)
  for i = n_start:2:length(v)-1
    v[i] = start_val
    v[i + 1] = start_val
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


"Convert the matrix indices `(i, j)` of `A ∈ R^{m x n}` into the corresponding linear index of `v = vec(A)`."
function mat_to_vec_ind(i::Int, j::Int, m::Int)
  (i > m || i <= 0 || j <= 0) && throw(BoundsError("Indices outside matrix bounds."))
  return (j - 1) * m + i
end


"Convert the matrix indices `(i, j)` of `A ∈ R^{m x n}` into the corresponding linear index of `v = svec(A, ::UpperTriangle)`."
function mat_to_svec_ind(i::Int, j::Int)
  if i <= j
    return div((j - 1) * j, 2) + i
  else
    return div((i - 1) * i, 2) + j
  end
end


vectorized_ind(i::Int, j::Int, m::Int, C::PsdConeTriangle{T}) where {T} = mat_to_svec_ind(i, j)
