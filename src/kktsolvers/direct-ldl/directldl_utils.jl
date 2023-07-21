using SparseArrays

struct LDLDataMap

    P::Vector{Int}
    A::Vector{Int}
    Hsblocks::Vector{Vector{Int}}  #indices of the lower RHS blocks (by cone)
    SOC_u::Vector{Vector{Int}}      #off diag dense columns u
    SOC_v::Vector{Vector{Int}}      #off diag dense columns v
    SOC_D::Vector{Int}              #diag of just the sparse SOC expansion D
    GenPow_p::Vector{Vector{Int}}   #off diag dense columns p
    GenPow_q::Vector{Vector{Int}}   #off diag dense columns q
    GenPow_r::Vector{Vector{Int}}   #off diag dense columns r
    GenPow_D::Vector{Int}           #diag of just the sparse GenPow expansion D
    PowM_p::Vector{Vector{Int}}   #off diag dense columns p
    PowM_q::Vector{Vector{Int}}   #off diag dense columns q
    PowM_r::Vector{Vector{Int}}   #off diag dense columns r
    PowM_D::Vector{Int}           #diag of just the sparse PowM expansion D

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

        #make an index for each of the Hs blocks for each cone
        Hsblocks = _allocate_kkt_Hsblocks(Int, cones)

<<<<<<< HEAD
        #now do the SOC expansion pieces and GenPow & PowerMean expansion pieces
        nsoc = cones.type_counts[Clarabel.SecondOrderConeT]
=======
        #now do the SOC expansion pieces
        nsoc = cones.type_counts[Clarabel.SecondOrderCone]
>>>>>>> original-git/main
        p    = 2*nsoc
        SOC_D = zeros(Int,p)
        SOC_u = Vector{Vector{Int}}(undef,nsoc)
        SOC_v = Vector{Vector{Int}}(undef,nsoc)

<<<<<<< HEAD
        n_genpow = cones.type_counts[Clarabel.GenPowerConeT]
        p_genpow = 3*n_genpow
        GenPow_p = Vector{Vector{Int}}(undef,n_genpow)
        GenPow_q = Vector{Vector{Int}}(undef,n_genpow)
        GenPow_r = Vector{Vector{Int}}(undef,n_genpow)
        GenPow_D = zeros(Int,p_genpow)   

        n_powm = cones.type_counts[Clarabel.PowerMeanConeT]
        p_powm = 3*n_powm
        PowM_p = Vector{Vector{Int}}(undef,n_powm)
        PowM_q = Vector{Vector{Int}}(undef,n_powm)
        PowM_r = Vector{Vector{Int}}(undef,n_powm)
        PowM_D = zeros(Int,p_powm)   

        count_soc = 1;
        count_genpow = 1;
        count_powm = 1;
        for (i,cone) in enumerate(cones)
            if isa(cones.cone_specs[i],Clarabel.SecondOrderConeT)
                SOC_u[count_soc] = Vector{Int}(undef,numel(cone))
                SOC_v[count_soc] = Vector{Int}(undef,numel(cone))
                count_soc += 1
            end
            if isa(cones.cone_specs[i],Clarabel.GenPowerConeT)
                GenPow_p[count_genpow] = Vector{Int}(undef,numel(cone))
                GenPow_q[count_genpow] = Vector{Int}(undef,cone.dim1)
                GenPow_r[count_genpow] = Vector{Int}(undef,cone.dim2)
                count_genpow += 1
            end
            if isa(cones.cone_specs[i],Clarabel.PowerMeanConeT)
                PowM_p[count_powm] = Vector{Int}(undef,numel(cone))
                PowM_q[count_powm] = Vector{Int}(undef,cone.d)
                PowM_r[count_powm] = Vector{Int}(undef,1)
                count_powm += 1
=======
        count = 1
        for cone in cones
            if isa(cone,Clarabel.SecondOrderCone)
                SOC_u[count] = Vector{Int}(undef,numel(cone))
                SOC_v[count] = Vector{Int}(undef,numel(cone))
                count += 1
>>>>>>> original-git/main
            end
        end

        diag_full = zeros(Int,m+n+p+p_genpow+p_powm)

        return new(P,A,Hsblocks,SOC_u,SOC_v,SOC_D,
                    GenPow_p,GenPow_q,GenPow_r,GenPow_D,
                    PowM_p,PowM_q,PowM_r,PowM_D,
                    diagP,diag_full)
    end

end

function _allocate_kkt_Hsblocks(type::Type{T}, cones) where{T <: Real}

    ncones    = length(cones)
    Hsblocks = Vector{Vector{T}}(undef,ncones)

    for (i, cone) in enumerate(cones)
        nvars = numel(cone)
        if Hs_is_diagonal(cone) 
            numelblock = nvars
        elseif isa(cones.cone_specs[i],Clarabel.EntropyConeT)
            numelblock = 2*nvars - 1 + cone.d
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

    (m,n)  = (size(A,1), size(P,1))
    n_socs = cones.type_counts[Clarabel.SecondOrderCone]
    p = 2*n_socs
    n_genpow = cones.type_counts[Clarabel.GenPowerConeT]
    p_genpow = 3*n_genpow
    n_powm = cones.type_counts[Clarabel.PowerMeanConeT]
    p_powm = 3*n_powm

    maps = LDLDataMap(P,A,cones)

    #entries actually on the diagonal of P
    nnz_diagP  = _count_diagonal_entries(P)

    # total entries in the Hs blocks
    nnz_Hsblocks = mapreduce(length, +, maps.Hsblocks; init = 0)

    #entries in the dense columns u/v of the
    #sparse SOC expansion terms.  2 is for
    #counting elements in both columns
    # GenPow & PowerMean has three vectors p,q,r, but length(p) = length(q) + length(r)
    nnz_SOC_vecs = 2*mapreduce(length, +, maps.SOC_u; init = 0)
    nnz_GenPow_vecs = mapreduce(length, +, maps.GenPow_p; init = 0)
    nnz_GenPow_vecs += mapreduce(length, +, maps.GenPow_q; init = 0)
    nnz_GenPow_vecs += mapreduce(length, +, maps.GenPow_r; init = 0)
    nnz_PowM_vecs = mapreduce(length, +, maps.PowM_p; init = 0)
    nnz_PowM_vecs += mapreduce(length, +, maps.PowM_q; init = 0)
    nnz_PowM_vecs += mapreduce(length, +, maps.PowM_r; init = 0)

    #entries in the sparse SOC diagonal extension block
    nnz_SOC_ext = length(maps.SOC_D)
    nnz_GenPow_ext = length(maps.GenPow_D)
    nnz_PowM_ext = length(maps.PowM_D)

    nnzKKT = (nnz(P) +   # Number of elements in P
    n -                  # Number of elements in diagonal top left block
    nnz_diagP +          # remove double count on the diagonal if P has entries
    nnz(A) +             # Number of nonzeros in A
    nnz_Hsblocks +      # Number of elements in diagonal below A'
    nnz_SOC_vecs +       # Number of elements in sparse SOC off diagonal columns
    nnz_SOC_ext +         # Number of elements in diagonal of SOC extension
    nnz_GenPow_vecs +   # Number of elements in sparse GenPow off diagonal columns
    nnz_GenPow_ext  +   # Number of elements in diagonal of GenPow extension
    nnz_PowM_vecs +   # Number of elements in sparse PowM off diagonal columns
    nnz_PowM_ext      # Number of elements in diagonal of PowM extension
    )

    K = _csc_spalloc(T, m+n+p+p_genpow+p_powm, m+n+p+p_genpow+p_powm, nnzKKT)

    _kkt_assemble_colcounts(K,P,A,cones,m,n,p,p_genpow,p_powm,shape)        #YC:Get stuck here
    _kkt_assemble_fill(K,maps,P,A,cones,m,n,p,p_genpow,p_powm,shape)

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
    p_genpow,
    p_powm,
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

    #add the the Hs blocks in the lower right
    for (i,cone) = enumerate(cones)
        firstcol = cones.headidx[i] + n
        blockdim = numel(cone)
        if Hs_is_diagonal(cone)
            _csc_colcount_diag(K,firstcol,blockdim)
        elseif isa(cones.cone_specs[i],Clarabel.EntropyConeT)
            _csc_colcount_entropy(K,firstcol,blockdim,shape)
        else
            _csc_colcount_dense_triangle(K,firstcol,blockdim,shape)
        end
    end

    #count dense columns for each SOC, GenPow, PowM
    socidx = 1  #which SOC are we working on?

    for (i,cone) in enumerate(cones)
        if isa(cone,Clarabel.SecondOrderCone)

            #we will add the u and v columns for this cone
            nvars   = numel(cone)
            headidx = cones.headidx[i]

            #which column does v go into?
            col = m + n + 2*socidx - 1

            if shape == :triu
                _csc_colcount_colvec(K,nvars,headidx + n, col)   #v column
                _csc_colcount_colvec(K,nvars,headidx + n, col+1) #u column
            else #:tril
                _csc_colcount_rowvec(K,nvars,col,   headidx + n) #v row
                _csc_colcount_rowvec(K,nvars,col+1, headidx + n) #u row
            end

            socidx = socidx + 1
        end
    end
    #add diagonal block in the lower RH corner
    #to allow for the diagonal terms in SOC expansion
    _csc_colcount_diag(K,n+m+1,p)

    socidx = socidx - 1 #num of SOC

    genpowidx = 1   #which GenPow are we working on?
    for i in eachindex(cones)
        if isa(cones.cone_specs[i],Clarabel.GenPowerConeT)

            #we will add the p,q,r columns for this cone
            nvars   = numel(cones[i])
            dim1 = cones[i].dim1
            dim2 = cones[i].dim2
            headidx = cones.headidx[i]

            #which column does p go into?
            col = m + n + 2*socidx + 3*genpowidx - 2

            if shape == :triu
                _csc_colcount_colvec(K,dim1,headidx + n, col) #q column
                _csc_colcount_colvec(K,dim2,headidx + dim1, col+1) #r column
                _csc_colcount_colvec(K,nvars,headidx + n, col+2)   #p column
            else #:tril
                _csc_colcount_rowvec(K,dim1,col, headidx + n) #q row
                _csc_colcount_rowvec(K,dim2,col+1, headidx + n + dim1) #r row
                _csc_colcount_rowvec(K,nvars,col+2, headidx + n) #p row
            end

            genpowidx = genpowidx + 1
        end
    end

    #add diagonal block in the lower RH corner
    #to allow for the diagonal terms in GenPow expansion
    _csc_colcount_diag(K,n+m+p+1,p_genpow)

    genpowidx = genpowidx - 1 #num of GenPow

    powmidx = 1   #which PowM are we working on?
    for i in eachindex(cones)
        if isa(cones.cone_specs[i],Clarabel.PowerMeanConeT)

            #we will add the p,q,r columns for this cone
            nvars   = numel(cones[i])
            dim1 = cones[i].d
            dim2 = one(Int)
            headidx = cones.headidx[i]

            #which column does p go into?
            col = m + n + 2*socidx + 3*genpowidx + 3*powmidx - 2

            if shape == :triu
                _csc_colcount_colvec(K,dim1,headidx + n, col) #q column
                _csc_colcount_colvec(K,dim2,headidx + dim1, col+1) #r column
                _csc_colcount_colvec(K,nvars,headidx + n, col+2)   #p column
            else #:tril
                _csc_colcount_rowvec(K,dim1,col, headidx + n) #q row
                _csc_colcount_rowvec(K,dim2,col+1, headidx + n + dim1) #r row
                _csc_colcount_rowvec(K,nvars,col+2, headidx + n) #p row
            end

            powmidx = powmidx + 1
        end
    end

    #add diagonal block in the lower RH corner
    #to allow for the diagonal terms in PowM expansion
    _csc_colcount_diag(K,n+m+p+p_genpow+1,p_powm)

    powmidx = powmidx - 1 #num of PowMean

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
    p_genpow,
    p_powm,
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


    #add the the Hs blocks in the lower right
    for (i,cone) = enumerate(cones)
        firstcol = cones.headidx[i] + n
        blockdim = numel(cone)
        if Hs_is_diagonal(cone)
            _csc_fill_diag(K,maps.Hsblocks[i],firstcol,blockdim)
        elseif isa(cones.cone_specs[i],Clarabel.EntropyConeT)
            _csc_fill_entropy(K,maps.Hsblocks[i],firstcol,blockdim,shape)
        else
            _csc_fill_dense_triangle(K,maps.Hsblocks[i],firstcol,blockdim,shape)
        end
    end

    #fill in dense columns for each SOC and GenPow & PowM
    socidx = 1  #which SOC are we working on?
    genpowidx = 1  #which GenPow are we working on?
    powmidx = 1  #which PowM are we working on?

    for (i,cone) in enumerate(cones)
        if isa(cone,Clarabel.SecondOrderCone)

            nvars = numel(cone)
            headidx = cones.headidx[i]

            #which column does v go into (if triu)?
            col = m + n + 2*socidx - 1

            #fill structural zeros for u and v columns for this cone
            #note v is the first extra row/column, u is second
            if shape == :triu
                _csc_fill_colvec(K, maps.SOC_v[socidx], headidx + n, col    ) #v
                _csc_fill_colvec(K, maps.SOC_u[socidx], headidx + n, col + 1) #u
            else #:tril
                _csc_fill_rowvec(K, maps.SOC_v[socidx], col    , headidx + n) #v
                _csc_fill_rowvec(K, maps.SOC_u[socidx], col + 1, headidx + n) #u
            end

            socidx += 1
        end
    end
    #fill in SOC diagonal extension with diagonal of structural zeros
    _csc_fill_diag(K,maps.SOC_D,n+m+1,p)

    socidx = socidx - 1 #num of SOC

    for i in eachindex(cones)
        if isa(cones.cone_specs[i],Clarabel.GenPowerConeT)

            nvars = numel(cones[i])
            dim1 = cones[i].dim1
            headidx = cones.headidx[i]

            #which column does p go into (if triu)?
            col = m + n + 2*socidx + 3*genpowidx - 2

            #fill structural zeros for p,q,r columns for this cone
            if shape == :triu
                _csc_fill_colvec(K, maps.GenPow_q[genpowidx], headidx + n, col)     #q 
                _csc_fill_colvec(K, maps.GenPow_r[genpowidx], headidx + n + dim1, col + 1) #r 
                _csc_fill_colvec(K, maps.GenPow_p[genpowidx], headidx + n, col + 2) #p
            else #:tril
                _csc_fill_rowvec(K, maps.GenPow_q[genpowidx], col, headidx + n)     #q
                _csc_fill_rowvec(K, maps.GenPow_r[genpowidx], col + 1, headidx + n + dim1) #r
                _csc_fill_rowvec(K, maps.GenPow_p[genpowidx], col + 2, headidx + n) #p
            end

            genpowidx += 1
        end
    end

    #fill in GenPow diagonal extension with diagonal of structural zeros
    _csc_fill_diag(K,maps.GenPow_D,n+m+p+1,p_genpow)

    genpowidx = genpowidx - 1 #num of GenPow

    for i in eachindex(cones)
        if isa(cones.cone_specs[i],Clarabel.PowerMeanConeT)

            nvars = numel(cones[i])
            dim1 = cones[i].d
            headidx = cones.headidx[i]

            #which column does p go into (if triu)?
            col = m + n + 2*socidx + 3*genpowidx + 3*powmidx - 2

            #fill structural zeros for p,q,r columns for this cone
            if shape == :triu
                _csc_fill_colvec(K, maps.PowM_q[powmidx], headidx + n, col)     #q 
                _csc_fill_colvec(K, maps.PowM_r[powmidx], headidx + n + dim1, col + 1) #r 
                _csc_fill_colvec(K, maps.PowM_p[powmidx], headidx + n, col + 2) #p
            else #:tril
                _csc_fill_rowvec(K, maps.PowM_q[powmidx], col, headidx + n)     #q
                _csc_fill_rowvec(K, maps.PowM_r[powmidx], col + 1, headidx + n + dim1) #r
                _csc_fill_rowvec(K, maps.PowM_p[powmidx], col + 2, headidx + n) #p
            end

            powmidx += 1
        end
    end

    #fill in GenPow diagonal extension with diagonal of structural zeros
    _csc_fill_diag(K,maps.PowM_D,n+m+p+p_genpow+1,p_powm)

    powmidx = powmidx - 1 #num of PowM

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
