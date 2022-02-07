import QDLDL

# ---------------
# KKT Solver (direct method)
# ---------------

mutable struct DefaultKKTSolverDirect{T} <: AbstractKKTSolver{T}

    n  #cols in A
    m  #rows in A
    p  #extra KKT columns for sparse SOCs

    KKT::SparseMatrixCSC{T}
    factors
    perm

    # a vector for storing the scaling
    # matrix diagonal entries
    diagW2::SplitVector{T}

    #solution vector for constant part of KKT solves
    lhs_cb::Vector{T}
    #views into constant solution LHS
    lhs_cb_x::VectorView{T}
    lhs_cb_z::VectorView{T}
    lhs_cb_p::VectorView{T}

    #RHS vector for constant part of KKT solves
    rhs_cb::Vector{T}
    #views into constant solution LHS
    rhs_cb_x::VectorView{T}
    rhs_cb_z::VectorView{T}
    rhs_cb_p::VectorView{T}

    #work vector for other solutions and its views
    work::Vector{T}
    work_x::VectorView{T}
    work_z::VectorView{T}
    work_p::VectorView{T}


    function DefaultKKTSolverDirect{T}(
        data::DefaultProblemData{T},
        scalings::DefaultScalings) where {T}

        n = data.n
        m = data.m
        p = 2*data.cone_info.k_socone

        #KKT, factors = initialize_kkt_matrix(data)
        #PJG: this function is ropey AF
        KKT,factors,perm = _initialize_kkt_matrix(data.P,data.A,n,m,p)

        #a vector for storing diagonal
        #terms of the scaling matrix
        diagW2 = SplitVector{T}(data.cone_info)

        #the LHS for the constant part of the reduced
        #solve and its solution
        lhs_cb = Vector{T}(undef,n + m + p)
        #views into the RHS/LHS for x/z partition
        lhs_cb_x = view(lhs_cb,1:n)
        lhs_cb_z = view(lhs_cb,(n+1):(n+m))
        lhs_cb_p = view(lhs_cb,(n+m+1):(n+m+p))

        #direct solver solves in place, so hold a copy
        #of the RHS built from constant data terms
        rhs_cb = Vector{T}(undef,n + m + p)
        rhs_cb_x = view(rhs_cb,1:n)
        rhs_cb_z = view(rhs_cb,(n+1):(n+m))
        rhs_cb_p = view(rhs_cb,(n+m+1):(n+m+p))

        rhs_cb_x .= -data.c;
        rhs_cb_z .=  data.b;
        rhs_cb_p .=  0.0;

        #work vector serving as both LHS and RHS
        #for linear system solves (solves in place)
        work  = Vector{T}(undef, n + m + p)

        #views into the work vector
        work_x = view(work,1:n)
        work_z = view(work,(n+1):(n+m))
        work_p = view(work,(n+m+1):(n+m+p))


        return new(
            n,m,p,KKT,nothing,nothing,diagW2,
            lhs_cb,lhs_cb_x,lhs_cb_z,lhs_cb_p,
            rhs_cb,rhs_cb_x,rhs_cb_z,rhs_cb_p,
            work, work_x, work_z, work_p)

    end

end

DefaultKKTSolverDirect(args...) = DefaultKKTSolverDirect{DefaultFloat}(args...)


function _initialize_kkt_matrix(P,A,n,m,p) where{T}

    #PJG: this is crazy inefficient
    D2  = sparse(I(m)*1.)
    D3  = sparse(I(p)*1.)
    ZA  = spzeros(m,n)
    KKT = [triu(P) A'; ZA D2]  #upper triangle only
    KKT = blockdiag(KKT,D3)
    factors = nothing
    perm    = nothing

    return KKT, factors, perm

end

function kkt_update!(
    kktsolver::DefaultKKTSolverDirect{T},
    scalings::DefaultScalings{T}
) where {T}

    n = kktsolver.n
    m = kktsolver.m
    p = kktsolver.p

    scaling_get_diagonal!(scalings,kktsolver.diagW2)

    #set the diagonal of the KKT matrix
    #PJG: this is super inefficient
    for i = 1:m
        kktsolver.KKT[(n+i),(n+i)] = kktsolver.diagW2.vec[i]
    end

    #add the scaled u and v columns.
    #only needed on the upper triangle
    colidx = n+1    #the first column of current cone
    pidx   = n+m+1  #next SOC expansion column goes here

    for i = 1:length(scalings.cone_info.types)

        conedim = scalings.cone_info.dims[i]

        if(scalings.cone_info.types[i] == SecondOrderConeT)

            K  = scalings.cones[i]
            η2 = K.η^2

            #add scaled u and v columns here
            rows = (colidx):(colidx+conedim-1)
            kktsolver.KKT[rows,pidx]   .= (-η2).*K.v
            kktsolver.KKT[rows,pidx+1] .= (-η2).*K.u

            #add 1/-1 to diagonal in the extended rows/cols
            kktsolver.KKT[pidx,pidx]      = -η2
            kktsolver.KKT[pidx+1,pidx+1]  = +η2
            pidx += 2
        end

        colidx += conedim

    end

    #PJG: permumation should be decided at
    #initialization, but compute it once here
    #instead until the KKT initialization is
    #properly placing sparse vectors on the borders
    if(isnothing(kktsolver.perm))
        kktsolver.perm = amd(kktsolver.KKT)
    end

    #refactor.  PJG: For now, just overwrite the factors
    kktsolver.factors = qdldl(kktsolver.KKT;perm=kktsolver.perm)

    #calculate KKT solution for constant terms
    _kkt_solve_constant_rhs!(kktsolver)

    return nothing
end


function _kkt_solve_constant_rhs!(
    kktsolver::DefaultKKTSolverDirect{T}
) where {T}

    # QDLDL solves in place, so we hold a copy [-c;b;0]
    # in the rhs_cb (built one time in the constructor)
    # and copy over for solve in place.  Additional zeros are
    # held at the of this vector for sparse SOC model
    kktsolver.lhs_cb .= kktsolver.rhs_cb

    QDLDL.solve!(kktsolver.factors,kktsolver.lhs_cb)

    return nothing
end

function kkt_solve_initial_point!(
    kktsolver::DefaultKKTSolverDirect{T},
    variables::DefaultVariables{T},
    data::DefaultProblemData{T}
) where{T}

    # solve with [0;b] as a RHS to get (x,s) initializers
    # zero out the sparse cone variables at end
    kktsolver.work_x .= 0.0
    kktsolver.work_z .= data.b
    kktsolver.work_p .= 0.0
    QDLDL.solve!(kktsolver.factors,kktsolver.work)
    variables.x      .= kktsolver.work_x
    variables.s.vec  .= kktsolver.work_z

    # solve with [-c;0] as a RHS to get z initializer
    kktsolver.work_x .= -data.c
    kktsolver.work_z .=  0.0
    kktsolver.work_p .=  0.0
    QDLDL.solve!(kktsolver.factors,kktsolver.work)
    variables.z.vec  .= kktsolver.work_z

    return nothing
end

function kkt_solve!(
    kktsolver::DefaultKKTSolverDirect{T},
    lhs::DefaultVariables{T},
    rhs::DefaultVariables{T},
    variables::DefaultVariables{T},
    scalings::DefaultScalings{T},
    data::DefaultProblemData{T}
) where{T}

    cones = scalings.cones

    #PJG: possible that phase is not needed since
    #warm starting solutions to newton steps
    #maybe doesn't make any sense

    constx = kktsolver.lhs_cb_x
    constz = kktsolver.lhs_cb_z

    # assemble the right hand side and solve in place
    kktsolver.work_x .= rhs.x
    kktsolver.work_z .= rhs.z.vec
    kktsolver.work_p .= 0
    QDLDL.solve!(kktsolver.factors,kktsolver.work)

    #copy back into the solution to get (Δx₂,Δz₂)
    lhs.x     .= kktsolver.work_x
    lhs.z.vec .= kktsolver.work_z

    #PJG: temporary wasteful of memory to compute stuff here
    ξ  = variables.x / variables.τ
    P  = data.P

    #solve for Δτ
    lhs.τ  = rhs.τ - rhs.κ/variables.τ + 2*dot(ξ,P,lhs.x) + dot(data.c,lhs.x) + dot(data.b,lhs.z.vec)
    lhs.τ /= variables.κ/variables.τ - dot(data.c,constx) - dot(data.b,constz)
             + dot(ξ - lhs.x,P,ξ - lhs.x) - dot(lhs.x,P,lhs.x)

    #PJG: NB: the denominator lhs.τ can be written in a nicer way, but it involves
    #the norm of Wz.   Leaving it this way for now

    #shift solution by pre-computed constant terms
    #to get (Δx, Δz) = (Δx₂,Δz₂) + Δτ(Δx₁,Δz₁)
    lhs.x     .+= lhs.τ .* constx
    lhs.z.vec .+= lhs.τ .* constz

    #solve for Δs = -Wᵀ(λ \ dₛ + WΔz)
    cones_inv_circle_op!(cones, lhs.s, scalings.λ, rhs.s) #Δs = λ \ dₛ
    cones_gemv_W!(cones, false, lhs.z, lhs.s,  1., 1.)    #Δs = WΔz + Δs
    cones_gemv_W!(cones,  true, lhs.s, lhs.s, -1., 0.0)   #Δs = -WᵀΔs

    #solve for Δκ
    lhs.κ      = -(rhs.κ + variables.κ * lhs.τ) / variables.τ

    return nothing
end
