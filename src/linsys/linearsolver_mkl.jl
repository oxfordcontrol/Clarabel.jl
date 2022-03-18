using Pardiso

# -------------------------------------
# MKL Pardiso linear solver
# -------------------------------------

mutable struct MKLPardisoLinearSolver{T} <: AbstractLinearSolver{T}

    # problem dimensions
    m::Int; n::Int; p::Int

    # Left and right hand sides for solves
    x::Vector{T}
    b::Vector{T}

    #KKT matrix and its permutation
    KKT::SparseMatrixCSC{T}

    #KKT mappings from problem data to KKT
    KKTmaps::KKTDataMaps

    #the pardiso solver
    ps::MKLPardisoSolver

    #the expected signs of D in LDL
    Dsigns::Vector{Int}

    # a vector for storing the block diagonal
    # WtW blocks in the KKT matrix
    WtWblocks::Vector{Vector{T}}

    #settings.   This just points back
    #to the main solver settings.  It
    #is not taking an internal copy
    settings::Settings{T}

    function MKLPardisoLinearSolver{T}(P,A,cones,m,n,settings) where {T}

        #solving in sparse format.  Need this many
        #extra variables for SOCs
        p = 2*cones.type_counts[SecondOrderConeT]

        #LHS/RHS/work for iterative refinement
        x    = Vector{T}(undef,n+m+p)
        b    = Vector{T}(undef,n+m+p)

        #MKL wants TRIU in CSR format, so we make TRIL in CSC format
        KKT, KKTmaps = _assemble_kkt_matrix(P,A,cones,:tril)

        #PJG: The Dsigns logic is repeated from the QDLDL
        #wrapper.   Should be consolidated
        #the expected signs of D in LDL
        Dsigns = ones(Int,n+m+p)
        Dsigns[n+1:n+m] .= -1

        #the trailing block of p entries should
        #have alternating signs
        sign = -1
        for idx = (n+m+1):(n+m+p)
            Dsigns[idx] = sign
            sign *= -1
        end

        if(settings.static_regularization_enable)
            @. KKT.nzval[KKTmaps.diagP] += settings.static_regularization_eps
        end

        #make our AMD ordering outside of the solver
        perm = amd(KKT)

        #make a pardiso object and perform logical factor
        ps = MKLPardisoSolver()
        set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
        pardisoinit(ps)
        fix_iparm!(ps, :N)
        set_phase!(ps, Pardiso.ANALYSIS)
        set_perm!(ps, perm)
        pardiso(ps, KKT, b)  #RHS is irrelevant

        #updates to the diagonal of KKT will be
        #assigned here before updating matrix entries
        WtWblocks = _allocate_kkt_WtW_blocks(T, cones)

        #we might (?) need to register a finalizer for the pardiso
        #object to free internal structures
        finalizer(ps -> set_phase!(ps, Pardiso.RELEASE_ALL), ps )

        return new(m,n,p,x,b,KKT,KKTmaps,ps,Dsigns,WtWblocks,settings)
    end

end

MKLPardisoLinearSolver(args...) = MKLPardisoLinearSolver{DefaultFloat}(args...)


function linsys_update!(
    linsys::MKLPardisoLinearSolver{T},
    cones::ConeSet{T}
) where {T}

    n = linsys.n
    m = linsys.m
    p = linsys.p
    settings = linsys.settings
    KKT     = linsys.KKT
    ps      = linsys.ps
    maps    = linsys.KKTmaps

    #Set the diagonal of the W^tW block in the KKT matrix.
    scaling_get_WtW_blocks!(cones,linsys.WtWblocks)
    for (index, values) in zip(maps.WtWblocks,linsys.WtWblocks)
        KKT.nzval[index] .= -values #change signs to get -W^TW
    end


    #PJG: Code is copied from QDLDL.   TODO: Consolidate.
    #Can't consolidate yet because QDLDL is carrying a
    #a copy of the KKT and updating both it an the one
    #that ends up internal to the QDLDL object
    #update the scaled u and v columns.
    cidx = 1        #which of the SOCs are we working on?

    for (i,K) = enumerate(cones)
        if(cones.types[i] == SecondOrderConeT)

                η2 = K.η^2

                KKT.nzval[maps.SOC_u[cidx]] .= (-η2).*K.u
                KKT.nzval[maps.SOC_v[cidx]] .= (-η2).*K.v

                #add η^2*(1/-1) to diagonal in the extended rows/cols
                KKT.nzval[maps.SOC_D[cidx*2-1]] = -η2
                KKT.nzval[maps.SOC_D[cidx*2  ]] = +η2

                cidx += 1
        end

    end

    #Perturb the diagonal terms WtW that we have just overwritten
    #with static regularizers.  Note that we don't want to shift
    #elements in the ULHS #(corresponding to P) since we already
    #shifted them at initialization and haven't overwritten it
    if(settings.static_regularization_enable)
        eps = settings.static_regularization_eps
        @. KKT.nzval[maps.diag_full] += eps*linsys.Dsigns
        @. KKT.nzval[maps.diagP]     -= eps
    end

    # Recompute the numeric factorization, using fake RHS
    set_phase!(ps, Pardiso.NUM_FACT)
    b = [1.]
    pardiso(ps, KKT, b)

    return nothing
end


function linsys_solve!(
    linsys::MKLPardisoLinearSolver{T},
    lhsx::Union{Nothing,AbstractVector{T}},
    lhsz::Union{Nothing,AbstractVector{T}}
) where {T}

    (x,b) = (linsys.x,linsys.b)

    ps  = linsys.ps
    KKT = linsys.KKT

    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    pardiso(ps, x, KKT, b)

    linsys_getlhs!(linsys,lhsx,lhsz)
    return nothing
end



function linsys_setrhs!(
    linsys::MKLPardisoLinearSolver{T},
    rhsx::AbstractVector{T},
    rhsz::AbstractVector{T}
) where {T}

    b = linsys.b
    (m,n,p) = (linsys.m,linsys.n,linsys.p)

    b[1:n]             .= rhsx
    b[(n+1):(n+m)]     .= rhsz
    b[(n+m+1):(n+m+p)] .= 0

    return nothing
end


function linsys_getlhs!(
    linsys::MKLPardisoLinearSolver{T},
    lhsx::Union{Nothing,AbstractVector{T}},
    lhsz::Union{Nothing,AbstractVector{T}}
) where {T}

    x = linsys.x
    (m,n,p) = (linsys.m,linsys.n,linsys.p)

    isnothing(lhsx) || (lhsx .= x[1:n])
    isnothing(lhsz) || (lhsz .= x[(n+1):(n+m)])

    return nothing
end
