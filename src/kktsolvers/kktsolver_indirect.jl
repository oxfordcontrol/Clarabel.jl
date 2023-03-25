using LinearOperators, IterativeSolvers
# -------------------------------------
# Generic Indirect KKTSolver 
# -------------------------------------

mutable struct IndirectKKTSolver{T} <: AbstractKKTSolver{T}

    # problem dimensions
    m::Int; n::Int; 

    # Left and right hand sides for solves
    x1::Vector{T}
    x2::ConicVector{T}
    b1::Vector{T}
    b2::ConicVector{T}

    # two work vectors are required for multiplying
    # through W and for intermediate products A*x
    work1::ConicVector{T}
    work2::ConicVector{T}

    # internal (shallow) copies of problem data.  
    # could be mapped here to some other format
    P::Symmetric{T,AbstractMatrix{T}}
    A::AbstractMatrix{T}

    # block diagonal data for the lower RHS 
    H::Vector{Vector{T}}

    #settings just points back to the main solver settings.
    #Required since there is no separate KKT settings container
    settings::Settings{T}


    function IndirectKKTSolver{T}(P,A,cones,m,n,settings) where {T}

        #LHS/RHS/work 
        x1    = Vector{T}(undef,n)
        b1    = Vector{T}(undef,n)
        x2    = ConicVector{T}(cones)
        b2    = ConicVector{T}(cones)
        work1 = ConicVector{T}(cones)
        work2 = ConicVector{T}(cones)

        #lower RHS block elements 
        #PJG: wiill only work for diagonal blocks
        nblocks = numel.(cones.cones)
        H = map(n -> zeros(T,n), nblocks)
  
        return new(m,n,x1,x2,b1,b2,
                   work1,work2,Symmetric(P),A,H,settings)
    end

end

IndirectKKTSolver(args...) = IndirectKKTSolver{DefaultFloat}(args...)

KKTSolversDict[:indirect] = IndirectKKTSolver


function kktsolver_update!(
    kktsolver::IndirectKKTSolver{T},
    cones::CompositeCone{T}
) where {T}

    get_Hs!(cones,kktsolver.H)

    #PJG: development optimism
    is_success = true
    return is_success
end


function kktsolver_setrhs!(
    kktsolver::IndirectKKTSolver{T},
    rhsx::AbstractVector{T},
    rhsz::AbstractVector{T}
) where {T}

    kktsolver.b1      .= rhsx
    kktsolver.b2.vec  .= rhsz

    return nothing
end


function kktsolver_getlhs!(
    kktsolver::IndirectKKTSolver{T},
    lhsx::Union{Nothing,AbstractVector{T}},
    lhsz::Union{Nothing,AbstractVector{T}}
) where {T}

    isnothing(lhsx) || (@views lhsx .= kktsolver.x1)
    isnothing(lhsz) || (@views lhsz .= kktsolver.x2.vec)

    return nothing
end


function kktsolver_solve!(
    kktsolver::IndirectKKTSolver{T},
    cones::CompositeCone{T},
    lhsx::Union{Nothing,AbstractVector{T}},
    lhsz::Union{Nothing,AbstractVector{T}}
) where {T}

    (P,A)  = (kktsolver.P, kktsolver.A)
    work1  = kktsolver.work1
    work2  = kktsolver.work2
    (x1,x2) = (kktsolver.x1,kktsolver.x2)
    (b1,b2) = (kktsolver.b1,kktsolver.b2)
    H       = kktsolver.H

    _indirect_solve_kkt(cones,x1,x2,P,A,H,b1,b2,work1,work2)

    #PJG: development optimism
    is_success = true

    if is_success
       kktsolver_getlhs!(kktsolver,lhsx,lhsz)
    end

    return is_success
end


function _indirect_solve_kkt(cones,x1,x2,P,A,H,b1,b2,work1,work2)

    # Here we should put our solver for the system 
    # [P     A'][x1] = [b1]
    # [A    -H ][x2]   [b2]
    #
    # I will assume here that :
    #
    # 1) we want to solve by condensing to a PSD form 
    # and doing an indirect solve of (P + A'*H^{-1}*A)x = r
    # 
    # 2) The matrix H = W^TW is symmetric, sign definite,
    # and diagonal (i.e. nonnegative cones only).  
    
    # Diagonal H is not really necessary since I only need 
    # to compute products H^-1*b for an indirect solver,
    # but in this prototype I have formed H and its inverse 
    # directly just for testing.
    #
    # for an actual indirect method based on condensing we only need to  
    # compute products y = H^{-1}b = (W^TW)^{-1}b.  For symmetric cones 
    #  we should be able to do something like:
    # 
    # mul_Winv!(cones,:T,work1,b2,one(T),zero(T))    #work1  = (W^T)^{-1}*b2 
    # mul_Winv!(cones,:N,work2,work1,one(T),zero(T)) #work2  = W^{-1}*work1
    #
    # At present the above won't compile because the CompositeCone 
    # container type only implements the general nonsymmetric cone
    # interface, i.e. no mul_W or mul_Winv is available without 
    # some hackery.

    # Here for simplicity I will instead assume that H is diagonal and only 
    # has one block (e.g. a single nonnegative cone constraint)

    @assert(length(H) == 1)

    H = Diagonal(H[1])
    Hinv = inv(H)

    # Solve (P + A'*H^{-1}*A)*b1 = x1.  Indirect method goes here 
    # Should produce same as ... 
    # x1 .= (P + A'*Hinv*A)\(b1 + A'*Hinv*b2);

    A    = LinearOperator(A);
    Hinv = LinearOperator(Hinv);
    P    = LinearOperator(P);
    M = P + (A'*Hinv*A);
    x1 .= cg(M,b1 + A'*(Hinv*b2.vec));

    # backsolve for x2. 
    x2 .= Hinv*(A*x1 - b2);

end


