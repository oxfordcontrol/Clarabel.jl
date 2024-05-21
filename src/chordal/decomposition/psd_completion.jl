# -----------------------------------
# psd completion
# -----------------------------------

# The psd entries of z that correspond to the zeros in s are not constrained by the problem.
# however, in order to make the dual psd cone positive semidefinite we have to do a
# positive semidefinite completion routine to choose the values

function psd_completion!(
    chordal_info::ChordalInfo{T}, 
    variables::AbstractVariables{T}
) where {T}

    # working now with the cones from the original
    # problem, not the decomposed ones 
    cones = chordal_info.init_cones

    # loop over psd cones
    row_ranges = collect(rng_cones_iterator(cones))

    # loop over just the patterns 
    for pattern in chordal_info.spatterns
        row_range = row_ranges[pattern.orig_index]
        z = @view variables.z[row_range]
        complete!(z, pattern)
    end

    return nothing
end

function complete!(z::AbstractVector{T},pattern::SparsityPattern) where{T}

    n = length(pattern.ordering)
    Z = zeros(T,n,n)
    svec_to_mat!(Z,z)
    psd_complete!(Z,pattern)
    mat_to_svec!(z,Z)

end 

# positive semidefinite completion (from Vandenberghe - Chordal Graphs..., p. 362)
# input: A - positive definite completable matrix
function psd_complete!(A::AbstractMatrix{T}, pattern::SparsityPattern) where {T <: AbstractFloat}

    sntree  = pattern.sntree
    p  = pattern.ordering
    ip = invperm(p)
    N  = size(A,2)

    # PJG: not clear if this copy of A is required, or 
    # whether I can operate directly on A by permuting the 
    # the indices in the loops below.  Only worth doing that 
    # if copying in or out of A is expensive.

    # permutate matrix based on ordering p
    # W is in the order that the cliques are based on
    W = A[p, p]

    # go through supernode tree in descending order (given a post-ordering). 
    # This is ensured in the get_snode, get_separators functions

    for j = (sntree.n_cliques - 1):-1:1

        # in order to obtain ν, α the vertex numbers of the supernode are 
        # mapped to the new position of the permuted matrix index
        # set of snd(i) sorted using the numerical ordering i,i+1,...i+ni
        ν = get_snode(sntree, j)

        # index set containing the elements of col(i) \ snd(i) 
        #sorted using numerical ordering σ(i)
        α = get_separators(sntree, j)

        # index set containing the row indices of the lower-triangular zeros in 
        # column i (i: representative index) sorted by σ(i)
        i = ν[1]

        ν = collect(ν) #unborks indexing below
        α = collect(α) #unborks indexing below

        η = collect(i+1:1:N)
        filter!(x -> !insorted(x, α) && !insorted(x, ν), η)

        Waa = W[α, α]
        Wαν = view(W, α, ν)
        Wηα = view(W, η, α)

        Y = zeros(length(α), length(ν))
        try
            Y[:, :] = Waa \ Wαν
        catch
            Waa_pinv = pinv(Waa)
            Y[:, :] = Waa_pinv * Wαν
        end

        W[η, ν] =  Wηα * Y

        # symmetry condition
        W[ν, η] = view(W, η, ν)'

    end

    # invert the permutation
    A[:, :] =  W[ip, ip]

end

