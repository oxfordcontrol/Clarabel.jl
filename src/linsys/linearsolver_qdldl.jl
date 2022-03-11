# -------------------------------------
# QDLDL linear solver
# -------------------------------------

mutable struct QDLDLLinearSolver{T} <: AbstractLinearSolver{T}

    # problem dimensions
    m
    n
    p

    # internal workspace for IR scheme
    work::Vector{T}

    #KKT matrix and its LDL factors
    KKT::SparseMatrixCSC{T}
    factors

    #KKT mappings from problem data to KKT
    KKTmaps::KKTDataMaps

    #symmetric view for residual calcs
    KKTsym::Symmetric{T, SparseMatrixCSC{T,Int64}}

    #the expected signs of D in LDL
    Dsigns::Vector{Int}

    # a vector for storing the diagonal entries
    # of the WtW block in the KKT matrix
    diagWtW::ConicVector{T}

    #settings.   This just points back
    #to the main solver settings.  It
    #is not taking an internal copy
    settings::Settings{T}

    function QDLDLLinearSolver{T}(P,A,cone_info,m,n,settings) where {T}

        #solving in sparse format.  Need this many
        #extra variables for SOCs
        p = 2*cone_info.type_counts[SecondOrderConeT]

        #iterative refinement work vector
        work = Vector{T}(undef,n+m+p)

        KKT, KKTmaps = _assemble_kkt_matrix(P,A,cone_info,:triu)

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
        diagWtW = ConicVector{T}(cone_info)

        return new(m,n,p,work,KKT,factors,KKTmaps,KKTsym,Dsigns,diagWtW,settings)
    end

end

QDLDLLinearSolver(args...) = QDLDLLinearSolver{DefaultFloat}(args...)


function linsys_is_soc_sparse_format(linsys::QDLDLLinearSolver{T}) where{T}
    return true
end

function linsys_soc_sparse_variables(linsys::QDLDLLinearSolver{T}) where{T}
    return linsys.p
end



function linsys_update!(
    linsys::QDLDLLinearSolver{T},
    scalings::DefaultScalings{T}
) where {T}

    n = linsys.n
    m = linsys.m
    p = linsys.p
    settings = linsys.settings
    KKT  = linsys.KKT
    F    = linsys.factors
    maps = linsys.KKTmaps

    scaling_get_diagonal!(scalings,linsys.diagWtW)

    #Set the diagonal of the W^tW block in the KKT matrix.
    #Note that we need to do this both for the KKT matrix
    #that we have constructed and also for the version
    #that is stored internally in our factorization.
    #PJG:
    #The former is needed for iterative refinement.  Maybe we
    #could get away without using it and just writing a
    #multiplication operator for the QDLDL object., or implement
    #iterative refinement directly with QDLDL
    update_values!(F,maps.diagWtW,linsys.diagWtW)
    KKT.nzval[maps.diagWtW] .= linsys.diagWtW

    #update the scaled u and v columns.
    cidx = 1        #which of the SOCs are we working on?

    for i = 1:length(scalings.cone_info.types)
        if(scalings.cone_info.types[i] == SecondOrderConeT)

                K  = scalings.cones[i]
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
        offset_values!(F,maps.diag_full,eps,linsys.Dsigns)
        offset_values!(F,maps.diagP,-eps)  #undo to the P shift

        #and the same for the KKT matrix we are still
        #relying on for the iterative refinement calc
        @. KKT.nzval[maps.diag_full] += eps*linsys.Dsigns
        @. KKT.nzval[maps.diagP]     -= eps
    end

    #refactor with new data
    refactor!(F)

    return nothing
end


function linsys_solve!(
    linsys::QDLDLLinearSolver{T},
    x::Vector{T},
    b::Vector{T}
) where {T}

    normb = norm(b,Inf)
    work  = linsys.work

    #iterative refinement params
    IR_enable    = linsys.settings.iterative_refinement_enable
    IR_reltol    = linsys.settings.iterative_refinement_reltol
    IR_abstol    = linsys.settings.iterative_refinement_abstol
    IR_maxiter   = linsys.settings.iterative_refinement_max_iter
    IR_stopratio = linsys.settings.iterative_refinement_stop_ratio

    #make an initial solve
    x .= b
    QDLDL.solve!(linsys.factors,x)

    if(!IR_enable); return nothing; end  #done

    #Note that K is only triu data, so need to
    #be careful when computing the residual here
    K      = linsys.KKT
    KKTsym = linsys.KKTsym
    lastnorme = Inf

    for i = 1:IR_maxiter

        #this is work = error = b - Kξ
        work .= b
        mul!(work,KKTsym,x,-1.,1.)
        norme = norm(work,Inf)

        # test for convergence before committing
        # to a refinement step
        if(norme <= IR_abstol + IR_reltol*normb)
            return nothing
        end

        #if we haven't improved by at least the halting
        #ratio since the last pass through, then abort
        if(lastnorme/norme < IR_stopratio)
            return nothing
        end

        #make a refinement and continue
        QDLDL.solve!(linsys.factors,work)     #this is Δξ
        x .+= work
    end

    return nothing
end
