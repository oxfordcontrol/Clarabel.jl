# ---------------
# KKT Solver
# ---------------

mutable struct DefaultKKTSolver{T} <: AbstractKKTSolver{T}

    n  #cols in A
    m  #rows in A
    p  #extra KKT columns for sparse SOCs

    #the linear solver engine
    linsys::AbstractLinearSolver{T}

    #solution vector for constant part of KKT solves
    lhs_const::Vector{T}
    lhs_const_x::VectorView{T}
    lhs_const_z::VectorView{T}
    lhs_const_p::VectorView{T}

    #solution vector for general KKT solves
    lhs::Vector{T}
    lhs_x::VectorView{T}
    lhs_z::VectorView{T}
    lhs_p::VectorView{T}

    #work vector for solves, e.g. right hand sides
    work::Vector{T}
    work_x::VectorView{T}
    work_z::VectorView{T}
    work_p::VectorView{T}


        function DefaultKKTSolver{T}(
            data::DefaultProblemData{T},
            scalings::DefaultScalings{T},
            settings::Settings{T},
            solvertype::DataType = QDLDLLinearSolver{T},
        ) where {T}

        #basic problem dimensions
        n = data.n
        m = data.m

        #create the linear solver
        linsys = solvertype(data.P,data.A,data.cone_info,m,n,settings)

        #does our solver use sparse SOC format?
        if linsys_is_soc_sparse_format(linsys)
            p = linsys_soc_sparse_variables(linsys)
        else
            p = 0
        end

        #the LHS constant part of the reduced solve
        lhs_const   = Vector{T}(undef,n + m + p)
        #views into the LHS for x/z partition
        lhs_const_x = view(lhs_const,1:n)
        lhs_const_z = view(lhs_const,(n+1):(n+m))
        lhs_const_p = view(lhs_const,(n+m+1):(n+m+p))

        #the LHS for other solves
        lhs      = Vector{T}(undef,n + m + p)
        #views into the LHS for x/z partition
        lhs_x    = view(lhs,1:n)
        lhs_z    = view(lhs,(n+1):(n+m))
        lhs_p    = view(lhs,(n+m+1):(n+m+p))

        #work vector for other solves
        work  = Vector{T}(undef, n + m + p)
        #views into the work vector
        work_x = view(work,1:n)
        work_z = view(work,(n+1):(n+m))
        work_p = view(work,(n+m+1):(n+m+p))


        return new(
            n,m,p,linsys,
            lhs_const,lhs_const_x,lhs_const_z,lhs_const_p,
            lhs,lhs_x,lhs_z,lhs_p,
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

    #make the RHS for the constant part
    #of the reduced solve
    kktsolver.work_x .= -data.q;
    kktsolver.work_z .=  data.b;
    kktsolver.work_p .=  0.0;

    linsys_solve!(kktsolver.linsys,kktsolver.lhs_const,kktsolver.work)

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

    linsys_solve!(kktsolver.linsys,kktsolver.lhs,kktsolver.work)
    variables.x      .=  kktsolver.lhs_x
    variables.s.vec  .= -kktsolver.lhs_z

    # solve with [-c;0] as a RHS to get z initializer
    # zero out any sparse cone variables at end
    kktsolver.work_x .= -data.q
    kktsolver.work_z .=  0.0
    kktsolver.work_p .=  0.0

    linsys_solve!(kktsolver.linsys,kktsolver.lhs,kktsolver.work)
    variables.z.vec  .= kktsolver.lhs_z

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
    constx = kktsolver.lhs_const_x
    constz = kktsolver.lhs_const_z

    # assemble the right hand side and solve
    kktsolver.work_x .= rhs.x
    kktsolver.work_z .= rhs.z.vec
    kktsolver.work_p .= 0

    linsys_solve!(kktsolver.linsys,kktsolver.lhs,kktsolver.work)

    #copy back into the solution to get (Δx₂,Δz₂)
    lhs.x     .= kktsolver.lhs_x
    lhs.z.vec .= kktsolver.lhs_z

    #use workx as scratch space now that lhs is copied
    ξ   = kktsolver.work_x
    ξ  .= variables.x / variables.τ
    P   = data.Psym

    #solve for Δτ
    tau_num = rhs.τ - rhs.κ/variables.τ + dot(data.q,lhs.x) + dot(data.b,lhs.z.vec) + 2*dot(ξ,P,lhs.x)

    #now offset ξ for the quadratic form in the denominator
    ξ_minus_x    = ξ   #alias to ξ, same as work_x
    ξ_minus_x  .-= lhs.x

    tau_den = (variables.κ/variables.τ - dot(data.q,constx) - dot(data.b,constz) + dot(ξ_minus_x,P,ξ_minus_x) - dot(lhs.x,P,lhs.x))

    # Δτ = tau_num/tau_den
    lhs.τ  = tau_num/tau_den

    #shift solution by pre-computed constant terms
    #to get (Δx, Δz) = (Δx₂,Δz₂) + Δτ(Δx₁,Δz₁)
    lhs.x     .+= lhs.τ .* constx
    lhs.z.vec .+= lhs.τ .* constz

    #solve for Δs = -Wᵀ(λ \ dₛ + WΔz)
    cones_inv_circle_op!(cones, lhs.s, scalings.λ, rhs.s) #Δs = λ \ dₛ
    cones_gemv_W!(cones, false, lhs.z, lhs.s,  1., 1.)    #Δs = WΔz + Δs

    #PJG: problem here.  Can't multiply in place so allocating memory
    #caution, trying to assign Δs = -WᵀΔs produces a bug
    tmp1_sv = deepcopy(lhs.z)
    tmp1_sv.vec .= lhs.s.vec
    cones_gemv_W!(cones,  true, tmp1_sv, lhs.s, -1., 0.0)   #Δs = -WᵀΔs

    #solve for Δκ
    lhs.κ = -(rhs.κ + variables.κ * lhs.τ) / variables.τ

    return nothing
end
