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

        #refinement workspace
        work = Vector{T}(undef,n+m+p)

        #PJG: partly building the KKT matrix here.
        #not properly including the W part yet
        KKT = _assemble_kkt_matrix(P,A,m,n,p)

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
        KKT = KKT + Diagonal(Dsigns).*1e-7

        factors = nothing
        perm    = nothing
        diagW2 = SplitVector{T}(cone_info)

        return new(m,n,p,work,KKT,factors,perm,Dsigns,diagW2,settings)
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
    for i = (n+1):(n+m+p)
        linsys.KKT[i,i] += linsys.Dsigns[i]*1e-7
    end

    #PJG: permutation should be decided at
    #initialization, but compute it once here
    #instead until the KKT initialization is
    #properly placing sparse vectors on the borders
    if(isnothing(linsys.perm))
        linsys.perm = amd(linsys.KKT)
    end

    #refactor.  PJG: For now, just overwrite the factors
    linsys.factors = qdldl(linsys.KKT;
                           perm=linsys.perm,
                           Dsigns = linsys.Dsigns)


    return nothing
end


function linsys_solve!(
    linsys::QDLDLLinearSolver{T},
    x::Vector{T},
    b::Vector{T}
) where {T}

    normb = norm(b)
    work  = linsys.work

    #make an initial solve
    x .= b
    QDLDL.solve!(linsys.factors,x)

    #PJG: Note that K is only triu, so need to
    #be careful when computing the residual here
    K = linsys.KKT
    Ksym = Symmetric(K)

    for i = 1:3
        work .= b - Ksym*x                    #this is e = b - Kξ
        QDLDL.solve!(linsys.factors,work)     #this is Δξ
        x .+= work
    end

    return nothing
end
