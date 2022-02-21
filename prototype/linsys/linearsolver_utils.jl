using SparseArrays

struct KKTDataMaps

    P::Vector{Int}
    A::Vector{Int}
    diagWtW::Vector{Int}            #diagonal of just the W^TW block
    SOC_u::Vector{Vector{Int}}      #off diag dense columns u
    SOC_v::Vector{Vector{Int}}      #off diag dense columns v
    SOC_D::Vector{Int}              #diag of just the sparse SOC expansion D

    #all of above terms should be disjoint and their union
    #should cover all of the uset data in the KKT matrix.  Now
    #we make two last redundant indices that will tell us where
    #the whole diagonal is, including structural zeros.

    diagP::Vector{Int}
    diag_full::Vector{Int}

    function KKTDataMaps(P,A,cone_info)

        m = size(A,1)
        n = size(P,1)

        P       = zeros(Int,nnz(P))
        A       = zeros(Int,nnz(A))
        diagWtW = zeros(Int,m)

        #the diagonal of the ULHS block P.
        #NB : we fill in structural zeros here even if the matrix
        #P is empty (e.g. as in an LP), so we can have entries in
        #index Pdiag that are not present in the index P
        diagP  = zeros(Int,n)

        #now do the SOC expansion pieces
        nsoc = cone_info.type_counts[IPSolver.SecondOrderConeT]
        p    = 2*nsoc
        SOC_D = zeros(Int,p)

        SOC_u = Vector{Vector{Int}}(undef,nsoc)
        SOC_v = Vector{Vector{Int}}(undef,nsoc)

        count = 1
        for i = 1:length(cone_info.dims)
            if(cone_info.types[i] == IPSolver.SecondOrderConeT)
                SOC_u[count] = Vector{Int}(undef,cone_info.dims[i])
                SOC_v[count] = Vector{Int}(undef,cone_info.dims[i])
                count = count+1
            end
        end

        diag_full = zeros(Int,m+n+p)

        return new(P,A,diagWtW,SOC_u,SOC_v,SOC_D,diagP,diag_full)
    end

end


function _assemble_kkt_matrix_fast(
    P::SparseMatrixCSC{T},
    A::SparseMatrixCSC{T},
    cone_info
) where{T}

    n = size(P,1)
    m = size(A,1)
    p = 2*cone_info.type_counts[IPSolver.SecondOrderConeT]

    ndiagP  = _count_diagonal_entries(P)

    #count entries in the dense columns u/v of the
    #sparse SOC expansion terms
    n_soc_vecs = 0
    for i = 1:length(cone_info.dims)
        if(cone_info.types[i] == IPSolver.SecondOrderConeT)
            n_soc_vecs += 2*cone_info.dims[i]
        end
    end

    nnzKKT = (P.colptr[n+1]-1 +   # Number of elements in P (-1 for one indexing)
    n -                           # Number of elements in diagonal top left block
    ndiagP +                      # remove double count on the diagonal if P has entries
    A.colptr[n+1]-1 +             # Number of nonzeros in A (-1 for one indexing)
    m +                           # Number of elements in diagonal below A'
    n_soc_vecs +                  # Number of elements in sparse SOC off diagonal columns
    p)                            # Number of elements in diagonal of SOC extension

    K    = _csc_spalloc(m+n+p, m+n+p, nnzKKT)
    maps = KKTDataMaps(P,A,cone_info)

    _kkt_assemble_csc(K,maps,P,A,cone_info,m,n,p)

    return K,maps

end


function _kkt_assemble_csc(
    K,
    maps,
    P,
    A,
    cone_info,
    m,
    n,
    p
)

    m = A.m
    n = P.n

    #use K.p to hold nnz entries in each
    #column of the KKT matrix
    K.colptr .= 0
    _kkt_colcount_block(K,P,1,false)
    _kkt_colcount_missing_diag(K,P,1)
    _kkt_colcount_block(K,A,n+1,true)
    _kkt_colcount_diag(K,n+1,m)

    #count dense columns for each SOC
    socidx = 1  #which SOC are we working on?

    for i = 1:length(cone_info.dims)
        if(cone_info.types[i] == IPSolver.SecondOrderConeT)

            #we will add the u and v columns for this cone
            conedim = cone_info.dims[i]

            #which column does u go into?
            col = m + n + 2*socidx - 1

            _kkt_colcount_colvec(K,conedim,col  ) #u column
            _kkt_colcount_colvec(K,conedim,col+1) #v column

            socidx = socidx + 1
        end
    end

    #add diagonal block in the lower RH corner
    #to allow for the diagonal terms in SOC expansion
    _kkt_colcount_diag(K,n+m+1,p)

    #cumsum total entries to convert to K.p
    _kkt_colcount_to_colptr(K)

    #fill in value for P, top left (columnwise)
    _kkt_fill_block(K,P,maps.P,1,1,false)
    _kkt_fill_missing_diag(K,P,1)  #after adding P, since triu form

    #fill in value for A, top right (transposed/rowwise)
    _kkt_fill_block(K,A,maps.A,1,n+1,true)

    #fill in lower right below A' with diagonal of structural zeros
    _kkt_fill_diag(K,maps.diagWtW,n+1,m)

    #fill in dense columns for each SOC
    socidx = 1  #which SOC are we working on?

    for i = 1:length(cone_info.dims)
        if(cone_info.types[i] == IPSolver.SecondOrderConeT)

            conedim = cone_info.dims[i]
            headidx = cone_info.headidx[i]

            #which column does u go into?
            col = m + n + 2*socidx - 1

            #fill structural zeros for u and v columns for this cone
            #note v is the first extra column, u is second
            _kkt_fill_colvec(K, maps.SOC_v[socidx], col    , headidx + n,conedim) #u
            _kkt_fill_colvec(K, maps.SOC_u[socidx], col + 1, headidx + n,conedim) #v

            socidx += 1
        end
    end

    #fill in SOC diagonal extension with diagonal of structural zeros
    _kkt_fill_diag(K,maps.SOC_D,n+m+1,p)

    #backshift the colptrs to recover K.p again
    _kkt_backshift_colptrs(K)

    #Now we can populate the index of the full diagonal.
    #We have filled in structural zeros on it everywhere
    #as needed and the matrix is triu, so the diagonal
    #entries are just index by the last element in each column
    maps.diag_full .= K.colptr[2:end] .- 1

    #and the diagonal of just the upper left
    maps.diagP     .= K.colptr[2:(n+1)] .- 1

end




function _csc_spalloc(m, n, nnz)

    colptr = zeros(Int,n+1)
    rowval = zeros(Int,nnz)
    nzval  = zeros(Float64,nnz)

    #set the final colptr entry to 1+nnz
    #Julia 1.7 constructor check fails without
    #this condition
    colptr[end] = nnz +  1

    return SparseMatrixCSC{Float64,Int64}(m,n,colptr,rowval,nzval)
end



#increment the K.colptr by the number of nonzeros
#in a square diagonal matrix placed on the diagonal.
#Used to increment, e.g. the lower RHS block diagonal
function _kkt_colcount_diag(K,initcol,blockcols)

    for i = initcol:(initcol + (blockcols - 1))
        K.colptr[i] += 1
    end
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
function _kkt_colcount_colvec(K,n,col)

    #just add the vector length to this column
    K.colptr[col] += n

end

#increment the K.colptr by the number of nonzeros in M
function _kkt_colcount_block(K,M,initcol,istranspose)

    if istranspose
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

#populate values from v using the K.colptr as indicator of
#next fill location in each row.  If v = nothing, populates
#entries with zeros
function _kkt_fill_colvec(K,vtoKKT,col,initrow,vlength)

    for i = 1:vlength
        dest           = K.colptr[col]
        K.rowval[dest] = initrow + i - 1
        K.nzval[dest]  = 0.
        vtoKKT[i]      = dest
        K.colptr[col] += 1
    end

end


#populate values from M using the K.colptr as indicator of
#next fill location in each row
function _kkt_fill_block(K,M,MtoKKT,initrow,initcol,istranspose)

    for i = 1:M.n
        for j = M.colptr[i]:(M.colptr[i+1]-1)
            if istranspose
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

#Populate the diagonal with 0s using the K.colptr as indicator of
#next fill location in each row
function _kkt_fill_diag(K,rhotoKKT,offset,blockdim)

    for i = 1:blockdim
        col                 = i + offset - 1
        dest                = K.colptr[col]
        K.rowval[dest]      = col
        K.nzval[dest]       = 0.  #structural zero
        K.colptr[col]      += 1
        rhotoKKT[i]         = dest
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
