# -------------------------------------
# QDLDL linear solver
# -------------------------------------

mutable struct QDLDLKKTSolver{T} <: AbstractKKTSolver{T}

    # problem dimensions
    m::Int; n::Int; p::Int

    # Left and right hand sides for solves
    x::Vector{T}
    b::Vector{T}

    # internal workspace for IR scheme
    work::Vector{T}

    #KKT matrix and its LDL factors
    KKT::SparseMatrixCSC{T}
    factors::QDLDL.QDLDLFactorisation{T, Int64}

    #KKT mappings from problem data to KKT
    KKTmaps::KKTDataMaps

    #symmetric view for residual calcs
    KKTsym::Symmetric{T, SparseMatrixCSC{T,Int64}}

    #the expected signs of D in LDL
    Dsigns::Vector{Int}

    # a vector for storing the block diagonal
    # WtW blocks in the KKT matrix
    WtWblocks::Vector{Vector{T}}

    #settings.   This just points back
    #to the main solver settings.  It
    #is not taking an internal copy
    settings::Settings{T}

    function QDLDLKKTSolver{T}(P,A,cones,m,n,settings) where {T}

        #solving in sparse format.  Need this many
        #extra variables for SOCs
        p = 2*cones.type_counts[SecondOrderConeT]

        #LHS/RHS/work for iterative refinement
        x    = Vector{T}(undef,n+m+p)
        b    = Vector{T}(undef,n+m+p)
        work = Vector{T}(undef,n+m+p)

        KKT, KKTmaps = _assemble_kkt_matrix(P,A,cones,:triu)

        #KKT will be triu data only, but we will want
        #the following to allow products like KKT*x
        KKTsym = Symmetric(KKT)

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


        #make a logical factorization to fix memory allocations
        factors = qdldl(
            KKT;
            Dsigns = Dsigns,
            regularize_eps   = settings.dynamic_regularization_eps,
            regularize_delta = settings.dynamic_regularization_delta,
            logical          = true
        )

        #updates to the diagonal of KKT will be
        #assigned here before updating matrix entries
        WtWblocks = _allocate_kkt_WtW_blocks(T, cones)

        return new(m,n,p,x,b,work,KKT,factors,KKTmaps,KKTsym,Dsigns,WtWblocks,settings)
    end

end

QDLDLKKTSolver(args...) = QDLDLKKTSolver{DefaultFloat}(args...)


function kktsolver_update!(
    kktsolver::QDLDLKKTSolver{T},
    cones::ConeSet{T}
) where {T}

    (m,n,p) = (kktsolver.m,kktsolver.n,kktsolver.p)

    settings = kktsolver.settings
    KKT  = kktsolver.KKT
    F    = kktsolver.factors
    maps = kktsolver.KKTmaps

    #Set the elements the W^tW blocks in the KKT matrix.
    #Note that we need to do this both for the KKT matrix
    #that we have constructed and also for the version
    #that is stored internally in our factorization.
    #PJG:
    #The former is needed for iterative refinement.  Maybe we
    #could get away without using it and just writing a
    #multiplication operator for the QDLDL object., or implement
    #iterative refinement directly with QDLDL
    scaling_get_WtW_blocks!(cones,kktsolver.WtWblocks)
    for (index, values) in zip(maps.WtWblocks,kktsolver.WtWblocks)
        #change signs to get -W^TW
        values .= -values
        update_values!(F,index,values)
        KKT.nzval[index] .= values
    end

    #update the scaled u and v columns.
    cidx = 1        #which of the SOCs are we working on?

    for (i,K) = enumerate(cones)
        if(cones.types[i] == SecondOrderConeT)

                η2 = K.η^2

                update_values!(F,maps.SOC_u[cidx],(-η2).*K.u)
                update_values!(F,maps.SOC_v[cidx],(-η2).*K.v)

                KKT.nzval[maps.SOC_u[cidx]] .= (-η2).*K.u
                KKT.nzval[maps.SOC_v[cidx]] .= (-η2).*K.v

                #add η^2*(1/-1) to diagonal in the extended rows/cols
                update_values!(F,[maps.SOC_D[cidx*2-1]],[-η2])
                update_values!(F,[maps.SOC_D[cidx*2  ]],[+η2])

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
        offset_values!(F,maps.diag_full,eps,kktsolver.Dsigns)
        offset_values!(F,maps.diagP,-eps)  #undo to the P shift

        #and the same for the KKT matrix we are still
        #relying on for the iterative refinement calc
        @. KKT.nzval[maps.diag_full] += eps*kktsolver.Dsigns
        @. KKT.nzval[maps.diagP]     -= eps
    end

    #refactor with new data
    refactor!(F)

    return nothing
end


function kktsolver_setrhs!(
    kktsolver::QDLDLKKTSolver{T},
    rhsx::AbstractVector{T},
    rhsz::AbstractVector{T}
) where {T}

    b = kktsolver.b
    (m,n,p) = (kktsolver.m,kktsolver.n,kktsolver.p)

    b[1:n]             .= rhsx
    b[(n+1):(n+m)]     .= rhsz
    b[(n+m+1):(n+m+p)] .= 0

    return nothing
end


function kktsolver_getlhs!(
    kktsolver::QDLDLKKTSolver{T},
    lhsx::Union{Nothing,AbstractVector{T}},
    lhsz::Union{Nothing,AbstractVector{T}}
) where {T}

    x = kktsolver.x
    (m,n,p) = (kktsolver.m,kktsolver.n,kktsolver.p)

    isnothing(lhsx) || (lhsx .= x[1:n])
    isnothing(lhsz) || (lhsz .= x[(n+1):(n+m)])

    return nothing
end


function kktsolver_solve!(
    kktsolver::QDLDLKKTSolver{T},
    lhsx::Union{Nothing,AbstractVector{T}},
    lhsz::Union{Nothing,AbstractVector{T}}
) where {T}

    (x,b,work) = (kktsolver.x,kktsolver.b,kktsolver.work)

    #make an initial solve
    x .= b
    QDLDL.solve!(kktsolver.factors,x)

    if(kktsolver.settings.iterative_refinement_enable)
        iterative_refinement(kktsolver)
    end

    kktsolver_getlhs!(kktsolver,lhsx,lhsz)
    return nothing
end


function iterative_refinement(kktsolver::QDLDLKKTSolver{T}) where{T}

    (x,b,work) = (kktsolver.x,kktsolver.b,kktsolver.work)

    #iterative refinement params
    IR_reltol    = kktsolver.settings.iterative_refinement_reltol
    IR_abstol    = kktsolver.settings.iterative_refinement_abstol
    IR_maxiter   = kktsolver.settings.iterative_refinement_max_iter
    IR_stopratio = kktsolver.settings.iterative_refinement_stop_ratio

    #Note that K is only triu data, so need to
    #be careful when computing the residual here
    K      = kktsolver.KKT
    KKTsym = kktsolver.KKTsym
    lastnorme = Inf

    normb = norm(b,Inf)

    for i = 1:IR_maxiter

        #this is work = error = b - Kξ
        work .= b
        mul!(work,KKTsym,x,-1.,1.)
        norme = norm(work,Inf)

        # test for convergence before committing
        # to a refinement step
        if(norme <= IR_abstol + IR_reltol*normb)
            break
        end

        #if we haven't improved by at least the halting
        #ratio since the last pass through, then abort
        if(lastnorme/norme < IR_stopratio)
            break
        end

        #make a refinement and continue
        QDLDL.solve!(kktsolver.factors,work)     #this is Δξ
        x .+= work
    end

    return nothing
end
