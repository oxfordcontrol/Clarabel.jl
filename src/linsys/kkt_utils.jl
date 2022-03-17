using SparseArrays

struct KKTDataMaps

    P::Vector{Int}
    A::Vector{Int}
    WtWblocks::Vector{Vector{Int}}  #indices of the lower RHS blocks (by cone)
    SOC_u::Vector{Vector{Int}}      #off diag dense columns u
    SOC_v::Vector{Vector{Int}}      #off diag dense columns v
    SOC_D::Vector{Int}              #diag of just the sparse SOC expansion D

    #all of above terms should be disjoint and their union
    #should cover all of the uset data in the KKT matrix.  Now
    #we make two last redundant indices that will tell us where
    #the whole diagonal is, including structural zeros.

    diagP::Vector{Int}
    diag_full::Vector{Int}

    function KKTDataMaps(P,A,cones)

        (m,n) = (size(A,1), size(P,1))
        P = zeros(Int,nnz(P))
        A = zeros(Int,nnz(A))

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

        count = 1
        for (i,cone) in enumerate(cones)
            if(cones.types[i] == Clarabel.SecondOrderConeT)
                SOC_u[count] = Vector{Int}(undef,numel(cone))
                SOC_v[count] = Vector{Int}(undef,numel(cone))
                count = count+1
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

    maps = KKTDataMaps(P,A,cones)

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

    _kkt_assemble_colcounts(K,maps,P,A,cones,m,n,p,shape)
    _kkt_assemble_fill(K,maps,P,A,cones,m,n,p,shape)

    return K,maps

end

function _kkt_assemble_colcounts(
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

    (m,n) = (A.m, P.n)

    #use K.p to hold nnz entries in each
    #column of the KKT matrix
    K.colptr .= 0

    if shape == :triu
        _kkt_colcount_block(K,P,1,:N)
        _kkt_colcount_missing_diag(K,P,1)
        _kkt_colcount_block(K,A,n+1,:T)
    else #:tril
        _kkt_colcount_missing_diag(K,P,1)
        _kkt_colcount_block(K,P,1,:T)
        _kkt_colcount_block(K,A,1,:N)
    end

    #add the the WtW blocks in the lower right
    for (i,cone) = enumerate(cones)
        firstcol = cones.headidx[i] + n
        blockdim = numel(cone)
        if WtW_is_diagonal(cone)
            _kkt_colcount_diag(K,firstcol,blockdim)
        else
            _kkt_colcount_dense_triangle(K,firstcol,blockdim,shape)
        end
    end

    #count dense columns for each SOC
    socidx = 1  #which SOC are we working on?

    for i = 1:length(cones)
        if(cones.types[i] == Clarabel.SecondOrderConeT)

            #we will add the u and v columns for this cone
            nvars   = numel(cones[i])
            headidx = cones.headidx[i]

            #which column does u go into?
            col = m + n + 2*socidx - 1

            if shape == :triu
                _kkt_colcount_colvec(K,nvars,headidx + n, col)   #u column
                _kkt_colcount_colvec(K,nvars,headidx + n, col+1) #v column
            else #:tril
                _kkt_colcount_rowvec(K,nvars,col,   headidx + n) #u row
                _kkt_colcount_rowvec(K,nvars,col+1, headidx + n) #v row
            end

            socidx = socidx + 1
        end
    end

    #add diagonal block in the lower RH corner
    #to allow for the diagonal terms in SOC expansion
    _kkt_colcount_diag(K,n+m+1,p)

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

    (m,n) = (A.m, P.n)

    #cumsum total entries to convert to K.p
    _kkt_colcount_to_colptr(K)

    if shape == :triu
        _kkt_fill_block(K,P,maps.P,1,1,:N)
        _kkt_fill_missing_diag(K,P,1)  #after adding P, since triu form
        #fill in value for A, top right (transposed/rowwise)
        _kkt_fill_block(K,A,maps.A,1,n+1,:T)
    else #:tril
        _kkt_fill_missing_diag(K,P,1)  #before adding P, since tril form
        _kkt_fill_block(K,P,maps.P,1,1,:T)
        #fill in value for A, bottom left (not transposed)
        _kkt_fill_block(K,A,maps.A,n+1,1,:N)
    end


    #add the the WtW blocks in the lower right
    for (i,cone) = enumerate(cones)
        firstcol = cones.headidx[i] + n
        blockdim = numel(cone)
        if WtW_is_diagonal(cone)
            _kkt_fill_diag(K,maps.WtWblocks[i],firstcol,blockdim)
        else
            _kkt_fill_dense_triangle(K,maps.WtWblocks[i],firstcol,blockdim,shape)
        end
    end

    #fill in dense columns for each SOC
    socidx = 1  #which SOC are we working on?

    for i = 1:length(cones)
        if(cones.types[i] == Clarabel.SecondOrderConeT)

            nvars = numel(cones[i])
            headidx = cones.headidx[i]

            #which column does u go into (if triu)?
            col = m + n + 2*socidx - 1

            #fill structural zeros for u and v columns for this cone
            #note v is the first extra row/column, u is second
            if shape == :triu
                _kkt_fill_colvec(K, maps.SOC_v[socidx], headidx + n, col,     nvars) #u
                _kkt_fill_colvec(K, maps.SOC_u[socidx], headidx + n, col + 1, nvars) #v
            else #:tril
                _kkt_fill_rowvec(K, maps.SOC_v[socidx], col    , headidx + n,nvars) #u
                _kkt_fill_rowvec(K, maps.SOC_u[socidx], col + 1, headidx + n,nvars) #v
            end

            socidx += 1
        end
    end

    #fill in SOC diagonal extension with diagonal of structural zeros
    _kkt_fill_diag(K,maps.SOC_D,n+m+1,p)

    #backshift the colptrs to recover K.p again
    _kkt_backshift_colptrs(K)

    #Now we can populate the index of the full diagonal.
    #We have filled in structural zeros on it everywhere.

    if shape == :triu
        #matrix is tril, so diagonal is first in each column
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


function _csc_spalloc(T::Type{<:AbstractFloat},m, n, nnz)

    colptr = zeros(Int,n+1)
    rowval = zeros(Int,nnz)
    nzval  = zeros(T,nnz)

    #set the final colptr entry to 1+nnz
    #Julia 1.7 constructor check fails without
    #this condition
    colptr[end] = nnz +  1

    return SparseMatrixCSC{T,Int64}(m,n,colptr,rowval,nzval)
end

#increment the K.colptr by the number of nonzeros
#in a dense upper/lower triangle on the diagonal.
function _kkt_colcount_dense_triangle(K,initcol,blockcols,shape)
    cols  = initcol:(initcol + (blockcols - 1))
    if shape === :triu
        K.colptr[cols] += 1:blockcols
    else
        K.colptr[cols] += blockcols:-1:1
    end
end

#increment the K.colptr by the number of nonzeros
#in a square diagonal matrix placed on the diagonal.
function _kkt_colcount_diag(K,initcol,blockcols)
    cols  = initcol:(initcol + (blockcols - 1))
    K.colptr[cols] .+= 1
end

#same as _kkt_count_diag, but counts places
#where the input matrix M has a missing
#diagonal entry.  M must be square and TRIU
function _kkt_colcount_missing_diag(K,M,initcol)

    for i = 1:M.n
        if((M.colptr[i] == M.colptr[i+1]) ||    #completely empty column
           (M.rowval[M.colptr[i+1]-1] != i)     #last element is not on diagonal
          )
            K.colptr[i + (initcol-1)] += 1
        end
    end
end

#increment the K.colptr by the a number of nonzeros.
#used to account for the placement of a column
#vector that partially populates the column
function _kkt_colcount_colvec(K,n,firstrow, firstcol)

    #just add the vector length to this column
    K.colptr[firstcol] += n

end

#increment the K.colptr by 1 for every element
#used to account for the placement of a column
#vector that partially populates the column
function _kkt_colcount_rowvec(K,n,firstrow,firstcol)

    #add one element to each of n consective columns
    #starting from initcol.  The row index doesn't
    #matter here.
    for i = 1:n
        K.colptr[firstcol + i - 1] += 1
    end

end

#increment the K.colptr by the number of nonzeros in M
#shape should be :N or :T (the latter for transpose)
function _kkt_colcount_block(K,M,initcol,shape::Symbol)

    if shape == :T
        nnzM = M.colptr[end]-1
        for i = 1:nnzM
            K.colptr[M.rowval[i] + (initcol - 1)] += 1
        end

    else
        #just add the column count
        for i = 1:M.n
            K.colptr[(initcol - 1) + i] += M.colptr[i+1]-M.colptr[i]
        end
    end
end

#populate a partial column with zeros using the K.colptr as indicator of
#next fill location in each row.
function _kkt_fill_colvec(K,vtoKKT,initrow,initcol,vlength)

    for i = 1:vlength
        dest               = K.colptr[initcol]
        K.rowval[dest]     = initrow + i - 1
        K.nzval[dest]      = 0.
        vtoKKT[i]          = dest
        K.colptr[initcol] += 1
    end

end

#populate a partial row with zeros using the K.colptr as indicator of
#next fill location in each row.
function _kkt_fill_rowvec(K,vtoKKT,initrow,initcol,vlength)

    for i = 1:vlength
        col            = initcol + i - 1
        dest           = K.colptr[col]
        K.rowval[dest] = initrow
        K.nzval[dest]  = 0.
        vtoKKT[i]      = dest
        K.colptr[col] += 1
    end

end


#populate values from M using the K.colptr as indicator of
#next fill location in each row.
#shape should be :N or :T (the latter for transpose)
function _kkt_fill_block(K,M,MtoKKT,initrow,initcol,shape)

    for i = 1:M.n
        for j = M.colptr[i]:(M.colptr[i+1]-1)
            if shape == :T
                col = M.rowval[j] + (initcol - 1)
                row = i + (initrow - 1)
            else
                col = i + (initcol - 1)
                row = M.rowval[j] + (initrow - 1)
            end
            dest           = K.colptr[col]
            K.rowval[dest] = row
            K.nzval[dest]  = M.nzval[j]
            MtoKKT[j]      = dest
            K.colptr[col] += 1
        end
    end
end

#Populate the upper or lower triangle with 0s using the K.colptr
#as indicator of next fill location in each row
function _kkt_fill_dense_triangle(K,blocktoKKT,offset,blockdim,shape)

    #data will always be supplied as triu, so when filling it into
    #a tril shape we also need to transpose it.   Just write two
    #separate cases for clarity here

    if(shape === :triu)
        kidx = 1
        for col in offset:(offset + blockdim - 1)
            for row in (offset:col)
                dest             = K.colptr[col]
                K.rowval[dest]   = row
                K.nzval[dest]    = 0.  #structural zero
                K.colptr[col]   += 1
                blocktoKKT[kidx] = dest
                kidx = kidx + 1
            end
        end

    else #shape ==== :tril
    kidx = 1
        for row in offset:(offset + blockdim - 1)
            for col in offset:row
                dest             = K.colptr[col]
                K.rowval[dest]   = row
                K.nzval[dest]    = 0.  #structural zero
                K.colptr[col]   += 1
                blocktoKKT[kidx] = dest
                kidx = kidx + 1
            end
        end
    end
end

#Populate the diagonal with 0s using the K.colptr as indicator of
#next fill location in each row
function _kkt_fill_diag(K,diagtoKKT,offset,blockdim)

    for i = 1:blockdim
        col                 = i + offset - 1
        dest                = K.colptr[col]
        K.rowval[dest]      = col
        K.nzval[dest]       = 0.  #structural zero
        K.colptr[col]      += 1
        diagtoKKT[i]        = dest
    end
end

#same as _kkt_fill_diag, but only places 0.
#entries where the input matrix M has a missing
#diagonal entry.  M must be square and TRIU
function _kkt_fill_missing_diag(K,M,initcol)

    for i = 1:M.n
        #fill out missing diagonal terms only
        if((M.colptr[i] == M.colptr[i+1]) ||    #completely empty column
           (M.rowval[M.colptr[i+1]-1] != i)     #last element is not on diagonal
          )
            dest           = K.colptr[i + (initcol - 1)]
            K.rowval[dest] = i + (initcol - 1)
            K.nzval[dest]  = 0.  #structural zero
            K.colptr[i]   += 1
        end
    end
end

function _kkt_colcount_to_colptr(K)

    currentptr = 1
    for i = 1:(K.n+1)
       count        = K.colptr[i]
       K.colptr[i]  = currentptr
       currentptr  += count
    end


end

function _kkt_backshift_colptrs(K)

    for i = K.n:-1:1
        K.colptr[i+1] = K.colptr[i]
    end
    K.colptr[1] = 1  #zero in C
end


function _count_diagonal_entries(P)

    count = 0
    i     = 0

    for i = 1:P.n

        #compare last entry in each column with
        #its row number to identify diagonal entries
        if((P.colptr[i+1] != P.colptr[i]) &&    #nonempty column
           (P.rowval[P.colptr[i+1]-1] == i) )   #last element is on diagonal
                count += 1
        end
    end
    return count

end
