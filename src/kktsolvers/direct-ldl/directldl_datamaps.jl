using SparseArrays, StaticArrays

abstract type SparseExpansionMap end 

pdim(maps::Vector{SparseExpansionMap}) = sum(pdim, maps; init = 0)
nnz_vec(maps::Vector{SparseExpansionMap}) = sum(nnz_vec, maps; init = 0)

struct SOCExpansionMap <: SparseExpansionMap
    u::Vector{DefaultInt}        #off diag dense columns u
    v::Vector{DefaultInt}        #off diag dense columns v
    D::MVector{2, DefaultInt}    #diag D
    function SOCExpansionMap(cone::SecondOrderCone)
        u = zeros(DefaultInt,numel(cone))
        v = zeros(DefaultInt,numel(cone))
        D = MVector(0,0)
        new(u,v,D)
    end
end
pdim(::SOCExpansionMap) = 2
nnz_vec(map::SOCExpansionMap) = 2*length(map.u)
Dsigns(::SOCExpansionMap) = (-1,1)
expansion_map(cone::SecondOrderCone) = SOCExpansionMap(cone)

function _csc_colcount_sparsecone(
    cone::SecondOrderCone,
    map::SOCExpansionMap,
    K::SparseMatrixCSC,
    row::DefaultInt,col::DefaultInt,shape::Symbol
)
    
    nvars = numel(cone)
    if shape == :triu
        _csc_colcount_colvec(K,nvars,row, col  ) #v column
        _csc_colcount_colvec(K,nvars,row, col+1) #u column
    else #:tril
        _csc_colcount_rowvec(K,nvars,col,   row) #v row
        _csc_colcount_rowvec(K,nvars,col+1, row) #u row
    end
    _csc_colcount_diag(K,col,pdim(map))
end

function _csc_fill_sparsecone(
    cone::SecondOrderCone,
    map::SOCExpansionMap,
    K::SparseMatrixCSC,row::DefaultInt,col::DefaultInt,shape::Symbol
)

    nvars = numel(cone)
    #fill structural zeros for u and v columns for this cone
    #note v is the first extra row/column, u is second
    if shape == :triu
        _csc_fill_colvec(K, map.v, row, col    ) #v
        _csc_fill_colvec(K, map.u, row, col + 1) #u
    else #:tril
        _csc_fill_rowvec(K, map.v, col    , row) #v
        _csc_fill_rowvec(K, map.u, col + 1, row) #u
    end
    _csc_fill_diag(K,map.D,col,pdim(map))
end 

function _csc_update_sparsecone(
    cone::SecondOrderCone{T},
    map::SOCExpansionMap, 
    updateFcn, 
    scaleFcn
) where {T}
    
    η2 = cone.η^2

    #off diagonal columns (or rows)
    updateFcn(map.u,cone.sparse_data.u)
    updateFcn(map.v,cone.sparse_data.v)
    scaleFcn(map.u,-η2)
    scaleFcn(map.v,-η2)

    #set diagonal to η^2*(-1,1) in the extended rows/cols
    updateFcn(map.D,[-η2,+η2])

end

struct GenPowExpansionMap <: SparseExpansionMap
    
    p::Vector{DefaultInt}        #off diag dense columns p
    q::Vector{DefaultInt}        #off diag dense columns q
    r::Vector{DefaultInt}        #off diag dense columns r
    D::MVector{3, DefaultInt}    #diag D

    function GenPowExpansionMap(cone::GenPowerCone)
        p = zeros(DefaultInt,numel(cone))
        q = zeros(DefaultInt,dim1(cone))
        r = zeros(DefaultInt,dim2(cone))
        D = MVector(0,0,0)
        new(p,q,r,D)
    end
end
pdim(::GenPowExpansionMap) = 3
nnz_vec(map::GenPowExpansionMap) = length(map.p) + length(map.q) + length(map.r)
Dsigns(::GenPowExpansionMap) = (-1,-1,+1)
expansion_map(cone::GenPowerCone) = GenPowExpansionMap(cone)

function _csc_colcount_sparsecone(
    cone::GenPowerCone,
    map::GenPowExpansionMap,
    K::SparseMatrixCSC,row::DefaultInt,col::DefaultInt,shape::Symbol
)

    nvars   = numel(cone)
    dim1 = Clarabel.dim1(cone)
    dim2 = Clarabel.dim2(cone)

    if shape == :triu
        _csc_colcount_colvec(K,dim1, row,        col)    #q column
        _csc_colcount_colvec(K,dim2, row + dim1, col+1)  #r column
        _csc_colcount_colvec(K,nvars,row,        col+2)  #p column
    else #:tril
        _csc_colcount_rowvec(K,dim1, col,   row)         #q row
        _csc_colcount_rowvec(K,dim2, col+1, row + dim1)  #r row
        _csc_colcount_rowvec(K,nvars,col+2, row)         #p row
    end
    _csc_colcount_diag(K,col,pdim(map))
    
end

function _csc_fill_sparsecone(
    cone::GenPowerCone{T},
    map::GenPowExpansionMap,
    K::SparseMatrixCSC{T},
    row::DefaultInt,col::DefaultInt,shape::Symbol
) where{T}

    dim1  = Clarabel.dim1(cone)

    if shape == :triu
        _csc_fill_colvec(K, map.q, row,        col)   #q
        _csc_fill_colvec(K, map.r, row + dim1, col+1) #r 
        _csc_fill_colvec(K, map.p, row,        col+2) #p 
    else #:tril
        _csc_fill_rowvec(K, map.q, col,   row)        #q
        _csc_fill_rowvec(K, map.r, col+1, row + dim1) #r
        _csc_fill_rowvec(K, map.p, col+2, row)        #p
    end
    _csc_fill_diag(K,map.D,col,pdim(map))

end 

function _csc_update_sparsecone(
    cone::GenPowerCone{T},
    map::GenPowExpansionMap, 
    updateFcn, 
    scaleFcn
) where {T}
    
    data  = cone.data
    sqrtμ = sqrt(data.μ)

    #off diagonal columns (or rows), distribute √μ to off-diagonal terms
    updateFcn(map.q,data.q)
    updateFcn(map.r,data.r)
    updateFcn(map.p,data.p)
    scaleFcn(map.q,-sqrtμ)
    scaleFcn(map.r,-sqrtμ)
    scaleFcn(map.p,-sqrtμ)

    #normalize diagonal terms to 1/-1 in the extended rows/cols
    updateFcn(map.D,[-one(T),-one(T),one(T)])
    
end


struct LDLDataMap

    P::Vector{DefaultInt}
    A::Vector{DefaultInt}
    Hsblocks::Vector{DefaultInt}                 #indices of the lower RHS blocks (by cone)
    sparse_maps::Vector{SparseExpansionMap}      #sparse cone expansion terms

    #all of above terms should be disjoint and their union
    #should cover all of the user data in the KKT matrix.  Now
    #we make two last redundant indices that will tell us where
    #the whole diagonal is, including structural zeros.
    diagP::Vector{DefaultInt}
    diag_full::Vector{DefaultInt}

    function LDLDataMap(Pmat::SparseMatrixCSC{T},Amat::SparseMatrixCSC{T},cones) where{T}

        (m,n) = (size(Amat,1), size(Pmat,1))
        P = zeros(DefaultInt,nnz(Pmat))
        A = zeros(DefaultInt,nnz(Amat))

        #the diagonal of the ULHS block P.
        #NB : we fill in structural zeros here even if the matrix
        #P is empty (e.g. as in an LP), so we can have entries in
        #index Pdiag that are not present in the index P
        diagP  = zeros(DefaultInt,n)

        #make an index for each of the Hs blocks for each cone
        Hsblocks = _allocate_kkt_Hsblocks(DefaultInt, cones)

        #now do the sparse cone expansion pieces
        nsparse = count(cone->(@conedispatch is_sparse_expandable(cone)),cones)
        sparse_maps = sizehint!(SparseExpansionMap[],nsparse)

        for cone in cones
            if @conedispatch is_sparse_expandable(cone) 
                push!(sparse_maps,expansion_map(cone))
            end
        end

        diag_full = zeros(DefaultInt,m+n+pdim(sparse_maps))

        return new(P,A,Hsblocks,sparse_maps,diagP,diag_full)
    end

end