using SparseArrays

struct LDLDataMap

    P::Vector{Int}
    A::Vector{Int}
    WtWblocks::Vector{Vector{Int}}  #indices of the lower RHS blocks (by cone)
    SOC_u::Vector{Vector{Int}}      #off diag dense columns u
    SOC_v::Vector{Vector{Int}}      #off diag dense columns v
    SOC_D::Vector{Int}              #diag of just the sparse SOC expansion D

    #all of above terms should be disjoint and their union
    #should cover all of the user data in the KKT matrix.  Now
    #we make two last redundant indices that will tell us where
    #the whole diagonal is, including structural zeros.

    diagP::Vector{Int}
    diag_full::Vector{Int}

    function LDLDataMap(Pmat,Amat,cones)

        (m,n) = (size(Amat,1), size(Pmat,1))
        P = zeros(Int,nnz(Pmat))
        A = zeros(Int,nnz(Amat))

        #the diagonal of the ULHS block P.
        #NB : we fill in structural zeros here even if the matrix
        #P is empty (e.g. as in an LP), so we can have entries in
        #index Pdiag that are not present in the index P
        diagP  = zeros(Int,n)

        #make an index for each of the WtW blocks for each cone
        WtWblocks = _allocate_kkt_WtW_blocks(Int, cones)

        #now do the SOC expansion pieces
        nsoc = cones.type_counts[Clarabel.SecondOrderConeT]
        p    = 2*nsoc
        SOC_D = zeros(Int,p)

        SOC_u = Vector{Vector{Int}}(undef,nsoc)
        SOC_v = Vector{Vector{Int}}(undef,nsoc)

        count = 1;
        for (i,cone) in enumerate(cones)
            if isa(cones.cone_specs[i],Clarabel.SecondOrderConeT)
                SOC_u[count] = Vector{Int}(undef,numel(cone))
                SOC_v[count] = Vector{Int}(undef,numel(cone))
                count += 1
            end
        end

        diag_full = zeros(Int,m+n+p)

        return new(P,A,WtWblocks,SOC_u,SOC_v,SOC_D,diagP,diag_full)
    end

end

function _allocate_kkt_WtW_blocks(type::Type{T}, cones) where{T <: Real}

    ncones    = length(cones)
    WtWblocks = Vector{Vector{T}}(undef,ncones)

    for (i, cone) in enumerate(cones)
        nvars = numel(cone)
        if WtW_is_diagonal(cone)
            numelblock = nvars
        else #dense triangle
            numelblock = (nvars*(nvars+1))>>1 #must be Int
        end
        WtWblocks[i] = Vector{T}(undef,numelblock)
    end

    return WtWblocks
end


function _assemble_kkt_matrix(
    P::SparseMatrixCSC{T},
    A::SparseMatrixCSC{T},
    cones::ConeSet{T},
    shape::Symbol = :triu  #or tril
) where{T}

    (m,n)  = (size(A,1), size(P,1))
    n_socs = cones.type_counts[Clarabel.SecondOrderConeT]
    p = 2*n_socs

    maps = LDLDataMap(P,A,cones)

    #entries actually on the diagonal of P
    nnz_diagP  = _count_diagonal_entries(P)

    # total entries in the WtW blocks
    nnz_WtW_blocks = mapreduce(length, +, maps.WtWblocks; init = 0)

    #entries in the dense columns u/v of the
    #sparse SOC expansion terms.  2 is for
    #counting elements in both columns
    nnz_SOC_vecs = 2*mapreduce(length, +, maps.SOC_u; init = 0)

    #entries in the sparse SOC diagonal extension block
    nnz_SOC_ext = length(maps.SOC_D)

    nnzKKT = (nnz(P) +   # Number of elements in P
    n -                  # Number of elements in diagonal top left block
    nnz_diagP +          # remove double count on the diagonal if P has entries
    nnz(A) +             # Number of nonzeros in A
    nnz_WtW_blocks +     # Number of elements in diagonal below A'
    nnz_SOC_vecs +       # Number of elements in sparse SOC off diagonal columns
    nnz_SOC_ext)         # Number of elements in diagonal of SOC extension

    K = _csc_spalloc(T, m+n+p, m+n+p, nnzKKT)

    _kkt_assemble_colcounts(K,P,A,cones,m,n,p,shape)
    _kkt_assemble_fill(K,maps,P,A,cones,m,n,p,shape)

    return K,maps

end

function _kkt_assemble_colcounts(
    K,
    P,
    A,
    cones,
    m,
    n,
    p,
    shape::Symbol
)

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

    #add the the WtW blocks in the lower right
    for (i,cone) = enumerate(cones)
        firstcol = cones.headidx[i] + n
        blockdim = numel(cone)
        if WtW_is_diagonal(cone)
            _csc_colcount_diag(K,firstcol,blockdim)
        else
            _csc_colcount_dense_triangle(K,firstcol,blockdim,shape)
        end
    end

    #count dense columns for each SOC
    socidx = 1  #which SOC are we working on?

    for i in eachindex(cones)
        if isa(cones.cone_specs[i],Clarabel.SecondOrderConeT)

            #we will add the u and v columns for this cone
            nvars   = numel(cones[i])
            headidx = cones.headidx[i]

            #which column does u go into?
            col = m + n + 2*socidx - 1

            if shape == :triu
                _csc_colcount_colvec(K,nvars,headidx + n, col)   #u column
                _csc_colcount_colvec(K,nvars,headidx + n, col+1) #v column
            else #:tril
                _csc_colcount_rowvec(K,nvars,col,   headidx + n) #u row
                _csc_colcount_rowvec(K,nvars,col+1, headidx + n) #v row
            end

            socidx = socidx + 1
        end
    end

    #add diagonal block in the lower RH corner
    #to allow for the diagonal terms in SOC expansion
    _csc_colcount_diag(K,n+m+1,p)

    return nothing
end


function _kkt_assemble_fill(
    K,
    maps,
    P,
    A,
    cones,
    m,
    n,
    p,
    shape::Symbol
)

    #cumsum total entries to convert to K.p
    _csc_colcount_to_colptr(K)

    if shape == :triu
        _csc_fill_block(K,P,maps.P,1,1,:N)
        _csc_fill_missing_diag(K,P,1)  #after adding P, since triu form
        #fill in value for A, top right (transposed/rowwise)
        _csc_fill_block(K,A,maps.A,1,n+1,:T)
    else #:tril
        _csc_fill_missing_diag(K,P,1)  #before adding P, since tril form
        _csc_fill_block(K,P,maps.P,1,1,:T)
        #fill in value for A, bottom left (not transposed)
        _csc_fill_block(K,A,maps.A,n+1,1,:N)
    end


    #add the the WtW blocks in the lower right
    for (i,cone) = enumerate(cones)
        firstcol = cones.headidx[i] + n
        blockdim = numel(cone)
        if WtW_is_diagonal(cone)
            _csc_fill_diag(K,maps.WtWblocks[i],firstcol,blockdim)
        else
            _csc_fill_dense_triangle(K,maps.WtWblocks[i],firstcol,blockdim,shape)
        end
    end

    #fill in dense columns for each SOC
    socidx = 1  #which SOC are we working on?

    for i in eachindex(cones)
        if isa(cones.cone_specs[i],Clarabel.SecondOrderConeT)

            nvars = numel(cones[i])
            headidx = cones.headidx[i]

            #which column does u go into (if triu)?
            col = m + n + 2*socidx - 1

            #fill structural zeros for u and v columns for this cone
            #note v is the first extra row/column, u is second
            if shape == :triu
                _csc_fill_colvec(K, maps.SOC_v[socidx], headidx + n, col    ) #u
                _csc_fill_colvec(K, maps.SOC_u[socidx], headidx + n, col + 1) #v
            else #:tril
                _csc_fill_rowvec(K, maps.SOC_v[socidx], col    , headidx + n) #u
                _csc_fill_rowvec(K, maps.SOC_u[socidx], col + 1, headidx + n) #v
            end

            socidx += 1
        end
    end

    #fill in SOC diagonal extension with diagonal of structural zeros
    _csc_fill_diag(K,maps.SOC_D,n+m+1,p)

    #backshift the colptrs to recover K.p again
    _kkt_backshift_colptrs(K)

    #Now we can populate the index of the full diagonal.
    #We have filled in structural zeros on it everywhere.

    if shape == :triu
        #matrix is triu, so diagonal is last in each column
        @views maps.diag_full .= K.colptr[2:end] .- 1
        #and the diagonal of just the upper left
        @views maps.diagP     .= K.colptr[2:(n+1)] .- 1

    else #:tril
        #matrix is tril, so diagonal is first in each column
        @views maps.diag_full .= K.colptr[1:end-1]
        #and the diagonal of just the upper left
        @views maps.diagP     .= K.colptr[1:n]
    end

    return nothing
end
