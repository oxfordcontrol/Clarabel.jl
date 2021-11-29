# ---------------
# KKT Solver
# ---------------

mutable struct DefaultKKTSolver{T} <: AbstractKKTSolver{T}
    n
    m
    KKT

    #solution vector for constant part of KKT solves
    lhs_cb::Vector{T}
    #views into constant solution LHS
    lhs_cb_x::VectorView{T}
    lhs_cb_z::VectorView{T}

    #work vector for other solutions and its views
    work::Vector{T}
    work_x::VectorView{T}
    work_z::VectorView{T}


    factors
    WtW   # PJG: temporary for debugging

    function DefaultKKTSolver{T}(data::DefaultProblemData{T}) where {T}

        Z   = spzeros(data.n,data.n)
        WtW = -sparse(I(data.m)*1.)
        A   = data.A
        KKT = [Z A'; A WtW]

        #work vector serving as both LHS and RHS
        #for linear system solves (solves in place)
        work  = Vector{T}(undef,data.n + data.m)

        #views into the work vector
        work_x = view(work,1:data.n)
        work_z = view(work,(data.n+1):(data.n+data.m))

        #the RHS/LHS for the constant part of the reduced
        #solve and its solution (solves in place)
        lhs_cb = Vector{T}(undef,data.n + data.m)

        #views into the LHS solution for x/z partition
        lhs_cb_x = view(lhs_cb,1:data.n)
        lhs_cb_z = view(lhs_cb,(data.n+1):(data.n+data.m))

        # PJG: need to allocate space here for factors
        # Left as nothing for now

        new(data.n,data.m,KKT,lhs_cb,lhs_cb_x,
            lhs_cb_z,work,work_x, work_z,nothing,nothing)
    end

end

DefaultKKTSolver(args...) = DefaultKKTSolver{DefaultFloat}(args...)




function UpdateKKTSystem!(kktsolver::DefaultKKTSolver{T},scalings::DefaultConeScalings{T}) where {T}

    #PJG : for now, just build the scaling matrix
    # and reconstruct KKT in some inefficient way
    WtW = make_scaling_matrix(scalings)

    #drop this block into the lower RHS of KKT
    kktsolver.KKT[(kktsolver.n+1):end,(kktsolver.n+1):end] .= -WtW

    #refactor.  For now, just overwrite the factors
    kktsolver.factors = qdldl(kktsolver.KKT)

    #PJG: remember the W block for debugging
    kktsolver.WtW = WtW

end


function SolveKKTConstantRHS!(kktsolver::DefaultKKTSolver{T},data::DefaultProblemData{T}) where {T}

    # QDLDL solves in place, so copy [-c;b] into it and solve
    # over to the solution vector
    kktsolver.lhs_cb_x .= -data.c;
    kktsolver.lhs_cb_z .=  data.b;
    solve!(kktsolver.factors,kktsolver.lhs_cb)

end

function SolveKKTInitialPoint!(kktsolver::DefaultKKTSolver{T},
                      variables::DefaultVariables{T},
                      data::DefaultProblemData{T}) where{T}

    # solve with [0;b] as a RHS to get (x,s) initializers
    kktsolver.work_x .= 0.
    kktsolver.work_z .= data.b
    solve!(kktsolver.factors,kktsolver.work)
    variables.x      .= kktsolver.work_x
    variables.s.vec  .= kktsolver.work_z

    # solve with [-c;-] as a RHS to get z initializer
    kktsolver.work_x .= -data.c
    kktsolver.work_z .=  0.
    solve!(kktsolver.factors,kktsolver.work)
    variables.z.vec  .= kktsolver.work_z

end

function SolveKKT!(kktsolver::DefaultKKTSolver{T},
                   lhs::DefaultVariables{T},
                   rhs::DefaultVariables{T},
                   variables::DefaultVariables{T},
                   scalings::DefaultConeScalings{T},
                   data::DefaultProblemData{T}) where{T}

    work   = kktsolver.work
    constx = kktsolver.lhs_cb_x
    constz = kktsolver.lhs_cb_z

    # assemble the right hand side and solve in place
    work[1:data.n]       .= rhs.x
    work[(data.n+1):end] .= rhs.z.vec
    solve!(kktsolver.factors,kktsolver.work)

    #copy back into the solution to get (Δx₂,Δz₂)
    lhs.x     .= work[1:data.n]
    lhs.z.vec .= work[(data.n+1):end]

    #solve for Δτ
    lhs.τ  = rhs.τ - rhs.κ/variables.τ + dot(data.c,lhs.x) + dot(data.b,lhs.z.vec)
    lhs.τ /= variables.κ/variables.τ - dot(data.c,constx) - dot(data.b,constz)

    #shift solution by pre-computed constant terms
    #to get (Δx, Δz) = (Δx₂,Δz₂) + Δτ(Δx₁,Δz₁)
    lhs.x     .+= lhs.τ .* constx
    lhs.z.vec .+= lhs.τ .* constz

    #solve for Δs = -Wᵀ(λ \ dₛ + WΔz)
    inv_circle_op!(scalings, lhs.s, variables.λ, rhs.s) #Δs = λ \ dₛ
    gemv_W!(scalings, false, lhs.z, lhs.s,  1., 1.)   #Δs = WΔz + Δs
    gemv_W!(scalings,  true, lhs.s, lhs.s, -1., 0.)   #Δs = -WᵀΔs

    #solve for Δκ
    lhs.κ      = -(rhs.κ + variables.κ * lhs.τ) / variables.τ

end
