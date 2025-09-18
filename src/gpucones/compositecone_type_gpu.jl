using CUDA, CUDA.CUSPARSE

# -------------------------------------
# collection of cones for composite
# operations on a compound set, including
# conewise scaling operations
# -------------------------------------

struct CompositeConeGPU{T} <: AbstractCone{T}
    #count of each cone type
    type_counts::Dict{Type,Cint}

    #overall size of the composite cone
    numel::Cint
    numel_linear::Cint
    degree::Cint

    #range views
    rng_cones::CuVector{UnitRange{Cint}}
    rng_blocks::CuVector{UnitRange{Cint}}

    # the flag for symmetric cone check
    _is_symmetric::Bool
    n_linear::Cint
    n_nn::Cint
    n_sparse_soc::Cint
    n_dense_soc::Cint
    n_soc::Cint
    n_exp::Cint
    n_pow::Cint
    n_psd::Cint

    idx_eq::Vector{Cint}
    idx_inq::Vector{Cint}

    #data
    w::CuVector{T}
    λ::CuVector{T}
    η::CuVector{T}

    #sparse SOC data
    d::CuVector{T}
    vut::CuVector{T}
    Matvut::CuSparseMatrix{T}

    #nonsymmetric cone
    αp::CuVector{T}           #power parameters of power cones
    H_dual::CuArray{T,3}            #Hessian of the dual barrier at z 
    Hs::CuArray{T,3}                #scaling matrix
    grad::CuMatrix{T}               #gradient of the dual barrier at z 

    #PSD cone
    psd_dim::Cint                  #We only support PSD cones with the same small dimension
    chol1::CuArray{T,3}
    chol2::CuArray{T,3}
    U::CuArray{T,3}
    S::CuMatrix{T}
    V::CuArray{T,3}    
    eigvals::CuMatrix{T}
    λpsd::CuMatrix{T}
    Λisqrt::CuMatrix{T}
    R::CuArray{T,3}
    Rinv::CuArray{T,3}
    Hspsd::CuArray{T,3}

    #step_size
    α::CuVector{T}

    #workspace for various internal uses
    workmat1::CuArray{T,3}
    workmat2::CuArray{T,3}
    workmat3::CuArray{T,3}
    workvec::CuVector{T}

    #workspace for sparse socs
    worksoc1::Union{CuVector{T}, Nothing}
    worksoc2::Union{CuVector{T}, Nothing}
    worksoc3::Union{CuVector{T}, Nothing}

    function CompositeConeGPU{T}(cone_specs::Vector{SupportedCone}, soc_threshold::Int) where {T}

        #Information from the CompositeCone on CPU 
        cone_orders = map(c -> orders(c, soc_threshold), cone_specs)
        #Guarantee the input cones are ordered
        if !issorted(cone_orders)
            error("The input cones should be ordered!")
        end
        cone_orders = nothing

        n_zero = Cint(count(x -> typeof(x) == ZeroConeT, cone_specs))
        n_nn = Cint(count(x -> typeof(x) == NonnegativeConeT, cone_specs))
        n_linear = n_zero + n_nn
        n_soc = Cint(count(x -> typeof(x) == SecondOrderConeT, cone_specs))
        n_exp = Cint(count(x -> typeof(x) == ExponentialConeT, cone_specs))
        n_pow = Cint(count(x -> typeof(x) == PowerConeT, cone_specs))
        n_psd = Cint(count(x -> typeof(x) == PSDTriangleConeT, cone_specs))

        type_counts = Dict{Type,DefaultInt}()
        if n_zero > 0
            type_counts[ZeroCone] = n_zero
        end
        if n_nn > 0
            type_counts[NonnegativeCone] = n_nn
        end
        if n_soc > 0
            type_counts[SecondOrderCone] = n_soc
        end
        if n_exp > 0
            type_counts[ExponentialCone] = n_exp
        end
        if n_pow > 0
            type_counts[PowerCone] = n_pow
        end
        if n_psd > 0
            type_counts[PSDTriangleCone] = n_psd
        end
        _is_symmetric = (n_exp + n_pow) > 0 ? false : true

        #idx set for eq and ineq constraints
        idx_eq = Vector{Cint}(undef, n_zero)
        idx_inq = Vector{Cint}(undef, n_nn)
        eq_i = zero(Cint)
        inq_i = zero(Cint)
        for i in 1:n_linear
            typeof(cone_specs[i]) === ZeroConeT ? idx_eq[eq_i+=1] = i : idx_inq[inq_i+=1] = i 
        end

        numel  = sum(cone -> nvars(cone), cone_specs; init = 0)
        degree = sum(cone -> degrees(cone), cone_specs; init = 0)

        #Generate ranges for cones
        rng_cones  = CuVector{UnitRange{Cint}}(collect(rng_cones_iterator(cone_specs)));
        rng_blocks = CuVector{UnitRange{Cint}}(collect(rng_blocks_iterator_full(cone_specs, soc_threshold)));

        @views numel_linear  = Cint(sum(cone -> nvars(cone), cone_specs[1:n_linear]; init = 0))
        @views numel_soc  = Cint(sum(cone -> nvars(cone), cone_specs[n_linear+1:n_linear+n_soc]; init = 0))

        w = CuVector{T}(undef,numel_linear+numel_soc)
        λ = CuVector{T}(undef,numel_linear+numel_soc)
        η = CuVector{T}(undef,n_soc)

        #Initialize space for nonsymmetric cones
        αp = Vector{T}(undef,n_pow)
        pow_ind = n_linear + n_soc + n_exp
        #store the power parameter of each power cone
        for i in 1:n_pow
            αp[i] = cone_specs[i+pow_ind].α
        end

        αp = CuVector(αp)
        H_dual = CuArray{T}(undef,3,3,n_exp+n_pow)
        Hs = CuArray{T}(undef,3,3,n_exp+n_pow)
        grad = CuArray{T}(undef,3,n_exp+n_pow)

        #PSD cone
        #We require all psd cones have the same dimensionality
        psd_ind = pow_ind + n_pow
        psd_dim = (n_psd > 0) ? cone_specs[psd_ind+1].dim : 0
        # for i in 1:n_psd
        #     if(psd_dim != cones[psd_ind+i].n)
        #         throw(DimensionMismatch("Not all positive definite cones have the same dimensionality!"))
        #     end
        # end
        @views @assert(all(cone -> cone.dim == psd_dim, cone_specs[psd_ind+1:psd_ind+n_psd]))

        chol1 = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        chol2 = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        U = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        S = CUDA.zeros(T,psd_dim,n_psd)
        V = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        eigvals = CUDA.zeros(T,psd_dim,n_psd)

        λpsd   = CUDA.zeros(T,psd_dim,n_psd)
        Λisqrt = CUDA.zeros(T,psd_dim,n_psd)
        R      = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        Rinv   = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        Hspsd  = CUDA.zeros(T,triangular_number(psd_dim),triangular_number(psd_dim),n_psd)

        workmat1 = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        workmat2 = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        workmat3 = CUDA.zeros(T,psd_dim,psd_dim,n_psd)
        workvec  = CUDA.zeros(T,triangular_number(psd_dim)*n_psd)

        α = CuVector{T}(undef, numel) #workspace for step size calculation and neighborhood check

        #Sparse second-order cone
        #idx set for sparse socs 
        @views n_sparse_soc = Cint(count(cone -> (nvars(cone) > soc_threshold), cone_specs[(n_linear+1):(n_linear+n_soc)]))
        n_dense_soc = n_soc - n_sparse_soc
        d = CuVector{T}(undef, n_sparse_soc)


        worksoc1 = n_sparse_soc > 0 ? CuVector{T}(undef, n_sparse_soc) : nothing
        worksoc2 = n_sparse_soc > 0 ? CuVector{T}(undef, n_sparse_soc) : nothing
        worksoc3 = n_sparse_soc > 0 ? CuVector{T}(undef, n_sparse_soc) : nothing

        @views numel_sparse_soc  = sum(cone -> nvars(cone), cone_specs[n_linear+1:n_linear+n_sparse_soc]; init = 0)
        vut = CuVector{T}(undef, 2*numel_sparse_soc)
        rowptr = CuVector{Cint}(undef, 2*n_sparse_soc+1)
        colval = CuVector{Cint}(undef, 2*numel_sparse_soc)

        #Initialize off-diagonals off sparse socs
        _sparse_soc_initialization_sequential!(rowptr, colval, rng_cones, numel_linear, n_linear, n_sparse_soc) 

        accumulate!(+, rowptr, rowptr)

        Matvut = CuSparseMatrixCSR{T}(rowptr, colval, vut, (2*n_sparse_soc, numel))

        return new(type_counts, numel, numel_linear, degree, rng_cones, rng_blocks, _is_symmetric,
                n_linear, n_nn, n_sparse_soc, n_dense_soc, n_soc, n_exp, n_pow, n_psd,
                idx_eq, idx_inq,
                w, λ, η, d, vut, Matvut,
                αp,H_dual,Hs,grad,
                psd_dim, chol1, chol2, U, S, V, eigvals, λpsd, Λisqrt, R, Rinv, Hspsd, α,
                workmat1, workmat2, workmat3, workvec,
                worksoc1, worksoc2, worksoc3)
    end
end

CompositeConeGPU(args...) = CompositeConeGPU{DefaultFloat}(args...)

Base.length(S::CompositeConeGPU{T}) where{T} = length(sum(values(S.type_counts)))

function get_type_count(cones::CompositeConeGPU{T}, type::DataType) where {T}
    typeT = ConeDict[type]
    if haskey(cones.type_counts,typeT)
        return cones.type_counts[typeT]
    else
        return 0
    end
end


# -------------------------------------
# iterators to generate indices into vectors 
# in a cone or cone-related blocks in the Hessian
struct RangeBlocksIteratorFull
    cones::Vector{SupportedCone}
    soc_threshold::Int
end

function rng_blocks_iterator_full(cones::Vector{SupportedCone}, soc_threshold::Int)
    RangeBlocksIteratorFull(cones, soc_threshold)
end

Base.length(iter::RangeBlocksIteratorFull) = length(iter.cones)

function Base.iterate(iter::RangeBlocksIteratorFull, state=(1, 1)) 
    (coneidx, start) = state 
    if coneidx > length(iter.cones)
        return nothing 
    else 
        cone = iter.cones[coneidx]
        nvars = Clarabel.nvars(cone)
        if (typeof(cone) == ZeroConeT || typeof(cone) == NonnegativeConeT || (typeof(cone) == SecondOrderConeT && nvars > iter.soc_threshold))
            stop = start + nvars - 1
        else
            stop = start + nvars*nvars - 1
        end
        state = (coneidx + 1, stop + 1)
        return (start:stop, state)
    end 
end 


# ------------------------------------
# initi offdiagonal terms for sparse socs
function _sparse_soc_initialization_sequential!(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    rng_cones::AbstractVector, 
    numel_linear::Cint, 
    n_linear::Cint, 
    n_sparse_soc::Cint
) 
    CUDA.@allowscalar begin
        rowptr[1] = one(Cint)
        for i in one(Cint):n_sparse_soc
            shift_i = i + n_linear
            rng_i = rng_cones[shift_i]
            len_i = Cint(length(rng_i))

            rowptr[2*i] = len_i
            rowptr[2*i+1] = len_i

            rng_sparse_i = rng_i .- numel_linear
            startidx = 2*(rng_sparse_i.stop - len_i)
            colvi = view(colval, (startidx+1):(startidx+len_i))
            colui = view(colval, (startidx+len_i+1):(startidx+2*len_i))
            copyto!(colvi, collect(rng_i))
            copyto!(colui, collect(rng_i))
        end
    end    
end

