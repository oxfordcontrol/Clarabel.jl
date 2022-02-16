function _assemble_kkt_matrix(P,A,m,n,p) where{T}

    #PJG: this is crazy inefficient
    D2  = sparse(I(m)*1.)
    D3  = sparse(I(p)*1.)
    ZA  = spzeros(m,n)

    #PJG : I will temporarily add 1e-300 here
    #just to force it to have diagonal entries
    #in its sparsity pattern.
    E   = I(n).*1e-300
    KKT = [triu(P + E) A'; ZA D2]  #upper triangle only
    KKT = blockdiag(KKT,D3)

    return KKT

end
