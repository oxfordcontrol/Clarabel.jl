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
    perm

    #symmetric view for residual calcs
    KKTsym::Symmetric{T, SparseMatrixCSC{T,Int64}}

    #the expected signs of D in LDL
    Dsigns::Vector{Int}

    # a vector for storing the scaling
    # matrix diagonal entries
    diagW2::SplitVector{T}

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

        #PJG: partly building the KKT matrix here.
        #not properly including the W part yet
        KKT = _assemble_kkt_matrix(P,A,m,n,p)

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

        #PJG:DEBUG
        #add dynamic regularization.   Just adds to the
        #diagonal here, and then subsequent modifications
        #to WtW should reapply regularization at the modified
        #entries only
        if(settings.static_regularization_enable)
            eps = settings.static_regularization_eps
            KKT .+= Diagonal(Dsigns).*eps
        end

        factors = nothing
        perm    = nothing
        diagW2 = SplitVector{T}(cone_info)

        return new(m,n,p,work,KKT,factors,perm,KKTsym,Dsigns,diagW2,settings)
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

    scaling_get_diagonal!(scalings,linsys.diagW2)

    #set the diagonal of the W^tW block in the KKT matrix
    #PJG: this is super inefficient
    for i = 1:m
        linsys.KKT[(n+i),(n+i)] = linsys.diagW2.vec[i]
    end

    #add off diagonal the scaled u and v columns.
    #only needed on the upper triangle
    colidx = n+1    #the first column of current cone
    pidx   = n+m+1  #next SOC expansion column goes here

    for i = 1:length(scalings.cone_info.types)

        conedim = scalings.cone_info.dims[i]

        if(scalings.cone_info.types[i] == SecondOrderConeT)

            K  = scalings.cones[i]
            η2 = K.η^2

            #add scaled u and v columns here
            #PJG: this is super inefficient
            rows = (colidx):(colidx+conedim-1)
            linsys.KKT[rows,pidx]   .= (-η2).*K.v
            linsys.KKT[rows,pidx+1] .= (-η2).*K.u

            #add 1/-1 to diagonal in the extended rows/cols
            linsys.KKT[pidx,pidx]      = -η2
            linsys.KKT[pidx+1,pidx+1]  = +η2
            pidx += 2
        end

        colidx += conedim

    end

    #perturb the diagonal terms that we are just overwritten
    #with a new version of WtW with static regularizers.
    #Note that we don't want to shift elements in the ULHS
    #(corresponding to P) since we already shifted them
    #at initialization and haven't overwritten it
    if(settings.static_regularization_enable)
        eps = settings.static_regularization_eps
        for i = (n+1):(n+m+p)
            linsys.KKT[i,i] += linsys.Dsigns[i]*eps
        end
    end

    #PJG: permutation should be decided at
    #initialization, but compute it once here
    #instead until the KKT initialization is
    #properly placing sparse vectors on the borders
    if(isnothing(linsys.perm))
        linsys.perm = amd(linsys.KKT)
    end

    #refactor.  PJG: For now, just overwrite the factors
    signs = settings.dynamic_regularization_enable ? linsys.Dsigns : nothing

    linsys.factors =qdldl(
            linsys.KKT;
            perm   = linsys.perm,
            Dsigns = linsys.Dsigns,
            regularize_eps   = settings.dynamic_regularization_eps,
            regularize_delta = settings.dynamic_regularization_delta
        )


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

    #PJG: Note that K is only triu data, so need to
    #be careful when computing the residual here
    K      = linsys.KKT
    KKTsym = linsys.KKTsym
    lastnorme = Inf

    for i = 1:IR_maxiter

        #this is work = error = b - Kξ
        work .= b
        mul!(work,KKTsym,x,1.,-1.)

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
