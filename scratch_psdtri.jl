function right_transform(n)

    D = triu(ones(n,n))*sqrt(2)
    D = D - Diagonal(D) + I(n)
    dD = Diagonal(D[:])
    Q = dD[findall(D[:] .!= 0),:]
    return Q

end

function left_transform(n)

    S = ones(n,n)*(1/sqrt(2))
    S= S - Diagonal(S) + I(n)

    A = triu(ones(n,n))
    A[:] = cumsum(A[:]) .* A[:]
    A = copy(Symmetric(A,:U))

    rows = collect(1:n^2)
    cols = A[:]
    vals = S[:]

    P = sparse(rows,cols,vals)

    return P

end

n = 3
K = Clarabel.PSDTriangleCone(n)
A = randsym(rng,n)
A[:] = cumsum(A[:]) .* A[:]
A = copy(Symmetric(A,:L))
a = zeros(K.numel)
vA = A[:]

Clarabel._tovec!(a,A,K)

Q = right_transform(n)
P = left_transform(n)
norm(Q*vA - a)
norm(vA - P*a)
