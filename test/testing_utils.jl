

function randsym(rng,n)
    A = randn(rng,n,n)
    A = A+A'
end


function randpsd(rng,n)
    A = randn(rng, n,n)
    A = A*A'
end
