# ---------------
# KKT Solver (direct method)
# ---------------

mutable struct DefaultKKTSolver{T} <: AbstractKKTSolver{T}

    n  #cols in A
    m  #rows in A
    p  #extra KKT columns for sparse SOCs

    #the linear solver engine
    linsys::AbstractLinearSolver{T}

    #solution vector for constant part of KKT solves
    lhs_cb::Vector{T}
    lhs_cb_x::VectorView{T}
    lhs_cb_z::VectorView{T}
    lhs_cb_p::VectorView{T}

    #work vector for other solutions and its views
    work::Vector{T}
    work_x::VectorView{T}
    work_z::VectorView{T}
    work_p::VectorView{T}


        function DefaultKKTSolver{T}(
            data::DefaultProblemData{T},
            scalings::DefaultScalings{T},
            solvertype::DataType = QDLDLLinearSolver{T}
        ) where {T}

        #basic problem dimensions
        n = data.n
        m = data.m

        #create the linear solver
        linsys = solvertype(data.P,data.A,data.cone_info,m,n)

        #does our solver use sparse SOC format?
        if linsys_is_soc_sparse_format(linsys)
            p = linsys_soc_sparse_variables(linsys)
        else
            p = 0
        end

        #the LHS for the constant part of the reduced
        #solve and its solution
        lhs_cb = Vector{T}(undef,n + m + p)
        #views into the RHS/LHS for x/z partition
        lhs_cb_x = view(lhs_cb,1:n)
        lhs_cb_z = view(lhs_cb,(n+1):(n+m))
        lhs_cb_p = view(lhs_cb,(n+m+1):(n+m+p))

        #work vector for other solves
        work  = Vector{T}(undef, n + m + p)
        #views into the work vector
        work_x = view(work,1:n)
        work_z = view(work,(n+1):(n+m))
        work_p = view(work,(n+m+1):(n+m+p))


        return new(
            n,m,p,linsys,
            lhs_cb,lhs_cb_x,lhs_cb_z,lhs_cb_p,
            work, work_x, work_z, work_p)

    end

end

DefaultKKTSolver(args...) = DefaultKKTSolver{DefaultFloat}(args...)


function kkt_update!(
    kktsolver::DefaultKKTSolver{T},
    data::DefaultProblemData{T},
    scalings::DefaultScalings{T}
) where {T}

    #update the linear solver with new scalings
    linsys_update!(kktsolver.linsys,scalings)

    #calculate KKT solution for constant terms
    _kkt_solve_constant_rhs!(kktsolver,data)

    return nothing
end


function _kkt_solve_constant_rhs!(
    kktsolver::DefaultKKTSolver{T},
    data::DefaultProblemData{T}
) where {T}

    kktsolver.lhs_cb_x .= -data.c;
    kktsolver.lhs_cb_z .=  data.b;
    kktsolver.lhs_cb_p .=  0.0;

    linsys_solve!(kktsolver.linsys,kktsolver.lhs_cb)

    return nothing
end


function kkt_solve_initial_point!(
    kktsolver::DefaultKKTSolver{T},
    variables::DefaultVariables{T},
    data::DefaultProblemData{T}
) where{T}

    # solve with [0;b] as a RHS to get (x,s) initializers
    # zero out any sparse cone variables at end
    kktsolver.work_x .= 0.0
    kktsolver.work_z .= data.b
    kktsolver.work_p .= 0.0
    linsys_solve!(kktsolver.linsys,kktsolver.work)
    variables.x      .= kktsolver.work_x
    variables.s.vec  .= kktsolver.work_z

    # solve with [-c;0] as a RHS to get z initializer
    # zero out any sparse cone variables at end
    kktsolver.work_x .= -data.c
    kktsolver.work_z .=  0.0
    kktsolver.work_p .=  0.0
    linsys_solve!(kktsolver.linsys,kktsolver.work)
    variables.z.vec  .= kktsolver.work_z

    return nothing
end


function kkt_solve!(
    kktsolver::DefaultKKTSolver{T},
    lhs::DefaultVariables{T},
    rhs::DefaultVariables{T},
    variables::DefaultVariables{T},
    scalings::DefaultScalings{T},
    data::DefaultProblemData{T}
) where{T}

    cones = scalings.cones
    constx = kktsolver.lhs_cb_x
    constz = kktsolver.lhs_cb_z

    # assemble the right hand side and solve in place
    kktsolver.work_x .= rhs.x
    kktsolver.work_z .= rhs.z.vec
    kktsolver.work_p .= 0
    linsys_solve!(kktsolver.linsys,kktsolver.work)

    #copy back into the solution to get (Δx₂,Δz₂)
    lhs.x     .= kktsolver.work_x
    lhs.z.vec .= kktsolver.work_z

    #PJG: temporary wasteful of memory to compute stuff here
    ξ  = variables.x / variables.τ
    P  = data.P

    #solve for Δτ
    lhs.τ  = + rhs.τ - rhs.κ/variables.τ
             + 2*dot(ξ,P,lhs.x) + dot(data.c,lhs.x)
             + dot(data.b,lhs.z.vec)

    lhs.τ /= + variables.κ/variables.τ - dot(data.c,constx)
             - dot(data.b,constz) + dot(ξ - lhs.x,P,ξ - lhs.x)
             - dot(lhs.x,P,lhs.x)

    #PJG: NB: the denominator lhs.τ can be written in a nicer way,
    #but it involves the norm of Wz.   Leaving it this way for now

    #shift solution by pre-computed constant terms
    #to get (Δx, Δz) = (Δx₂,Δz₂) + Δτ(Δx₁,Δz₁)
    lhs.x     .+= lhs.τ .* constx
    lhs.z.vec .+= lhs.τ .* constz

    #solve for Δs = -Wᵀ(λ \ dₛ + WΔz)
    cones_inv_circle_op!(cones, lhs.s, scalings.λ, rhs.s) #Δs = λ \ dₛ
    cones_gemv_W!(cones, false, lhs.z, lhs.s,  1., 1.)    #Δs = WΔz + Δs
    cones_gemv_W!(cones,  true, lhs.s, lhs.s, -1., 0.0)   #Δs = -WᵀΔs

    #solve for Δκ
    lhs.κ = -(rhs.κ + variables.κ * lhs.τ) / variables.τ

    return nothing
end
