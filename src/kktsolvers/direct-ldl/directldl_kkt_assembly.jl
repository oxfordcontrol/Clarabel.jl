using SparseArrays, StaticArrays

function _allocate_kkt_Hsblocks(type::Type{T}, cones) where{T <: Real}

    ncones    = length(cones)
    Hsblocks = Vector{Vector{T}}(undef,ncones)

    for (i, cone) in enumerate(cones)
        nvars = numel(cone)
        if Hs_is_diagonal(cone) 
            numelblock = nvars
        else #dense triangle
            numelblock = triangular_number(nvars) #must be Int
        end
        Hsblocks[i] = Vector{T}(undef,numelblock)
    end

    return Hsblocks
end


function _assemble_kkt_matrix(
    P::SparseMatrixCSC{T},
    A::SparseMatrixCSC{T},
    cones::CompositeCone{T},
    shape::Symbol = :triu  #or tril
) where{T}

    map   = LDLDataMap(P,A,cones)
    (m,n) = (size(A,1), size(P,1))
    p     = pdim(map.sparse_maps)

    #entries actually on the diagonal of P
    nnz_diagP  = _count_diagonal_entries(P)

    # total entries in the Hs blocks
    nnz_Hsblocks = mapreduce(length, +, map.Hsblocks; init = 0)

    nnzKKT = (nnz(P) +      # Number of elements in P
    n -                     # Number of elements in diagonal top left block
    nnz_diagP +             # remove double count on the diagonal if P has entries
    nnz(A) +                # Number of nonzeros in A
    nnz_Hsblocks +          # Number of elements in diagonal below A'
    nnz_vec(map.sparse_maps) + # Number of elements in sparse cone off diagonals
    p                       # Number of elements in diagonal of sparse cones
    )

    K = _csc_spalloc(T, m+n+p, m+n+p, nnzKKT)

    _kkt_assemble_colcounts(K,P,A,cones,map,shape)       
    _kkt_assemble_fill(K,P,A,cones,map,shape)

    return K,map

end

function _kkt_assemble_colcounts(
    K::SparseMatrixCSC{T},
    P::SparseMatrixCSC{T},
    A::SparseMatrixCSC{T},
    cones,
    map,
    shape::Symbol
) where{T}

    (m,n) = size(A)

    #use K.p to hold nnz entries in each
    #column of the KKT matrix
    K.colptr .= 0

    if shape == :triu
        _csc_colcount_block(K,P,1,:N)
        _csc_colcount_missing_diag(K,P,1)
        _csc_colcount_block(K,A,n+1,:T)
    else #:tril
        _csc_colcount_missing_diag(K,P,1)
        _csc_colcount_block(K,P,1,:T)
        _csc_colcount_block(K,A,1,:N)
    end

    # track the next sparse column to fill (assuming triu fill)
    pcol = m + n + 1 #next sparse column to fill
    sparse_map_iter = Iterators.Stateful(map.sparse_maps)
    
    for (i,cone) = enumerate(cones)
        row = cones.headidx[i] + n

        #add the the Hs blocks in the lower right
        blockdim = numel(cone)
        if Hs_is_diagonal(cone)
            _csc_colcount_diag(K,row,blockdim)
        else
            _csc_colcount_dense_triangle(K,row,blockdim,shape)
        end

        #add sparse expansions columns for sparse cones 
        if @conedispatch is_sparse_expandable(cone)  
            thismap = popfirst!(sparse_map_iter)
            _csc_colcount_sparsecone(cone,thismap,K,row,pcol,shape)
            pcol += pdim(thismap) #next sparse column to fill 
        end 
    end

    return nothing
end


function _kkt_assemble_fill(
    K::SparseMatrixCSC{T},
    P::SparseMatrixCSC{T},
    A::SparseMatrixCSC{T},
    cones,
    map,
    shape::Symbol
) where{T}

    (m,n) = size(A)

    #cumsum total entries to convert to K.p
    _csc_colcount_to_colptr(K)

    if shape == :triu
        _csc_fill_block(K,P,map.P,1,1,:N)
        _csc_fill_missing_diag(K,P,1)  #after adding P, since triu form
        #fill in value for A, top right (transposed/rowwise)
        _csc_fill_block(K,A,map.A,1,n+1,:T)
    else #:tril
        _csc_fill_missing_diag(K,P,1)  #before adding P, since tril form
        _csc_fill_block(K,P,map.P,1,1,:T)
        #fill in value for A, bottom left (not transposed)
        _csc_fill_block(K,A,map.A,n+1,1,:N)
    end

    # track the next sparse column to fill (assuming triu fill)
    pcol = m + n + 1 #next sparse column to fill
    sparse_map_iter = Iterators.Stateful(map.sparse_maps)

    for (i,cone) = enumerate(cones)
        row = cones.headidx[i] + n

        #add the the Hs blocks in the lower right
        blockdim = numel(cone)

        if Hs_is_diagonal(cone)
            _csc_fill_diag(K,map.Hsblocks[i],row,blockdim)
        else
            _csc_fill_dense_triangle(K,map.Hsblocks[i],row,blockdim,shape)
        end

        #add sparse expansions columns for sparse cones 
        if @conedispatch is_sparse_expandable(cone) 
            thismap = popfirst!(sparse_map_iter)
            _csc_fill_sparsecone(cone,thismap,K,row,pcol,shape)
            pcol += pdim(thismap) #next sparse column to fill 
        end 
    end
    
    #backshift the colptrs to recover K.p again
    _kkt_backshift_colptrs(K)

    #Now we can populate the index of the full diagonal.
    #We have filled in structural zeros on it everywhere.

    if shape == :triu
        #matrix is triu, so diagonal is last in each column
        @views map.diag_full .= K.colptr[2:end] .- 1
        #and the diagonal of just the upper left
        @views map.diagP     .= K.colptr[2:(n+1)] .- 1

    else #:tril
        #matrix is tril, so diagonal is first in each column
        @views map.diag_full .= K.colptr[1:end-1]
        #and the diagonal of just the upper left
        @views map.diagP     .= K.colptr[1:n]
    end

    return nothing
end
