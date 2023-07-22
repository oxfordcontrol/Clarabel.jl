using SparseArrays

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
function _csc_colcount_dense_triangle(K,initcol,blockcols,shape)
    cols  = initcol:(initcol + (blockcols - 1))
    if shape === :triu
        @views K.colptr[cols] += 1:blockcols
    else
        @views K.colptr[cols] += blockcols:-1:1
    end
end


#increment the K.colptr by the number of nonzeros
#in a square diagonal matrix placed on the diagonal.
function _csc_colcount_diag(K,initcol,blockcols)
    cols  = initcol:(initcol + (blockcols - 1))
    @views K.colptr[cols] .+= 1
end

#same as _kkt_count_diag, but counts places
#where the input matrix M has a missing
#diagonal entry.  M must be square and TRIU
function _csc_colcount_missing_diag(K,M,initcol)

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
function _csc_colcount_colvec(K,n,firstrow, firstcol)

    #just add the vector length to this column
    K.colptr[firstcol] += n

end

#increment the K.colptr by 1 for every element
#used to account for the placement of a column
#vector that partially populates the column
function _csc_colcount_rowvec(K,n,firstrow,firstcol)

    #add one element to each of n consective columns
    #starting from initcol.  The row index doesn't
    #matter here.
    for i = 1:n
        K.colptr[firstcol + i - 1] += 1
    end

end

#increment the K.colptr by the number of nonzeros in M
#shape should be :N or :T (the latter for transpose)
function _csc_colcount_block(K,M,initcol,shape::Symbol)

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
function _csc_fill_colvec(K,vtoKKT,initrow,initcol)

    for i = 1:length(vtoKKT)
        dest               = K.colptr[initcol]
        K.rowval[dest]     = initrow + i - 1
        K.nzval[dest]      = 0.
        vtoKKT[i]          = dest
        K.colptr[initcol] += 1
    end

end

#populate a partial row with zeros using the K.colptr as indicator of
#next fill location in each row.
function _csc_fill_rowvec(K,vtoKKT,initrow,initcol)

    for i = 1:length(vtoKKT)
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
function _csc_fill_block(K,M,MtoKKT,initrow,initcol,shape)

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
function _csc_fill_dense_triangle(K,blocktoKKT,offset,blockdim,shape)

    #data will always be supplied as triu, so when filling it into
    #a tril shape we also need to transpose it.   Just write two
    #separate cases for clarity here

    if(shape === :triu)
        _fill_dense_triangle_triu(K,blocktoKKT,offset,blockdim)
    else #shape ==== :tril
        _fill_dense_triangle_tril(K,blocktoKKT,offset,blockdim)
    end
end

function _fill_dense_triangle_triu(K,blocktoKKT,offset,blockdim)

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
end

function _fill_dense_triangle_tril(K,blocktoKKT,offset,blockdim)

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

#Populate the diagonal with 0s using the K.colptr as indicator of
#next fill location in each row
function _csc_fill_diag(K,diagtoKKT,offset,blockdim)

    for i = 1:blockdim
        col                 = i + offset - 1
        dest                = K.colptr[col]
        K.rowval[dest]      = col
        K.nzval[dest]       = 0.  #structural zero
        K.colptr[col]      += 1
        diagtoKKT[i]        = dest
    end
end

#same as _csc_fill_diag, but only places 0.
#entries where the input matrix M has a missing
#diagonal entry.  M must be square and TRIU
function _csc_fill_missing_diag(K,M,initcol)

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

function _csc_colcount_to_colptr(K)

    currentptr = 1
    for i = 1:(K.n+1)
       count        = K.colptr[i]
       K.colptr[i]  = currentptr
       currentptr  += count
    end


end

function _kkt_backshift_colptrs(K)

    #NB: julia circshift! not used since does not operate in place on a single vector, i.e. circshift!(a,a,1) is disallowed
    for i = K.n:-1:1
        K.colptr[i+1] = K.colptr[i]
    end
    K.colptr[1] = 1  #zero in C

end


function _count_diagonal_entries(P)

    count = 0
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
