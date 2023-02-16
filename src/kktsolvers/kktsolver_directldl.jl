# -------------------------------------
# KKTSolver using direct LDL factorisation
# -------------------------------------

mutable struct DirectLDLKKTSolver{T} <: AbstractKKTSolver{T}

    # problem dimensions
    m::Int; n::Int; p::Int

    # Left and right hand sides for solves
    x::Vector{T}
    b::Vector{T}

    # internal workspace for IR scheme
    # and static offsetting of KKT
    work1::Vector{T}
    work2::Vector{T}

    #KKT mapping from problem data to KKT
    map::LDLDataMap

    #the expected signs of D in KKT = LDL^T
    Dsigns::Vector{Int}

    # a vector for storing the Hs blocks
    # on the in the KKT matrix block diagonal
    Hsblocks::Vector{Vector{T}}

    #unpermuted KKT matrix
    KKT::SparseMatrixCSC{T,Int}

    #symmetric view for residual calcs
    KKTsym::Symmetric{T, SparseMatrixCSC{T,Int}}

    #settings just points back to the main solver settings.
    #Required since there is no separate LDL settings container
    settings::Settings{T}

    #the direct linear LDL solver
    ldlsolver::AbstractDirectLDLSolver{T}

    #the diagonal regularizer currently applied
    diagonal_regularizer::T


    function DirectLDLKKTSolver{T}(P,A,cones,m,n,settings) where {T}

        #solving in sparse format.  Need this many
        #extra variables for SOCs
        p = 2*cones.type_counts[SecondOrderConeT]

        #LHS/RHS/work for iterative refinement
        x    = Vector{T}(undef,n+m+p)
        b    = Vector{T}(undef,n+m+p)
        work_e  = Vector{T}(undef,n+m+p)
        work_dx = Vector{T}(undef,n+m+p)

        #the expected signs of D in LDL
        Dsigns = Vector{Int}(undef,n+m+p)
        _fill_Dsigns!(Dsigns,m,n,p)

        #updates to the diagonal of KKT will be
        #assigned here before updating matrix entries
        Hsblocks = _allocate_kkt_Hsblocks(T, cones)

        #which LDL solver should I use?
        ldlsolverT = _get_ldlsolver_type(settings.direct_solve_method)

        #does it want a :triu or :tril KKT matrix?
        kktshape = required_matrix_shape(ldlsolverT)
        KKT, map = _assemble_kkt_matrix(P,A,cones,kktshape)

        diagonal_regularizer = zero(T)

        #KKT will be triu data only, but we will want
        #the following to allow products like KKT*x
        KKTsym = Symmetric(KKT)

        #the LDL linear solver engine
        ldlsolver = ldlsolverT{T}(KKT,Dsigns,settings)

        return new(m,n,p,x,b,
                   work_e,work_dx,map,Dsigns,Hsblocks,
                   KKT,KKTsym,settings,ldlsolver,
                   diagonal_regularizer)
    end

end

DirectLDLKKTSolver(args...) = DirectLDLKKTSolver{DefaultFloat}(args...)

function _get_ldlsolver_type(s::Symbol)
    try
        return DirectLDLSolversDict[s]
    catch
        throw(error("Unsupported direct LDL linear solver :", s))
    end
end

function _fill_Dsigns!(Dsigns,m,n,p)

    Dsigns .= 1

    #flip expected negative signs of D in LDL
    Dsigns[n+1:n+m] .= -1

    #the trailing block of p entries should
    #have alternating signs
    Dsigns[(n+m+1):2:(n+m+p)] .= -1
end

#update entries in the kktsolver object using the
#given index into its CSC representation
function _update_values!(
    ldlsolver::AbstractDirectLDLSolver{T},
    KKT::SparseMatrixCSC{T,Ti},
    index::Vector{Ti},
    values::Vector{T}
) where{T,Ti}

    #Update values in the KKT matrix K
    _update_values_KKT!(KKT,index,values)

    #give the LDL subsolver an opportunity to update the same
    #values if needed.   This latter is useful for QDLDL
    #since it stores its own permuted copy
    update_values!(ldlsolver,index,values)

end

#updates KKT matrix values
function _update_values_KKT!(
    KKT::SparseMatrixCSC{T,Int},
    index::Vector{Ti},
    values::Vector{T}
) where{T,Ti}

    #Update values in the KKT matrix K
    @. KKT.nzval[index] = values

end

#scale entries in the kktsolver object using the
#given index into its CSC representation
function _scale_values!(
    ldlsolver::AbstractDirectLDLSolver{T},
    KKT::SparseMatrixCSC{T,Ti},
    index::Vector{Ti},
    scale::T
) where{T,Ti}

    #Update values in the KKT matrix K
    _scale_values_KKT!(KKT,index,scale)

    #give the LDL subsolver an opportunity to update the same
    #values if needed.   This latter is useful for QDLDL
    #since it stores its own permuted copy
    scale_values!(ldlsolver,index,scale)

end

#updates KKT matrix values
function _scale_values_KKT!(
    KKT::SparseMatrixCSC{T,Int},
    index::Vector{Ti},
    scale::T
) where{T,Ti}

    #Update values in the KKT matrix K
    @. KKT.nzval[index] *= scale

end


function kktsolver_update!(
    kktsolver::DirectLDLKKTSolver{T},
    cones::CompositeCone{T}
) where {T}

    # the internal ldlsolver is type unstable, so multiple
    # calls to the ldlsolvers will be very slow if called
    # directly.   Grab it here and then call an inner function
    # so that the ldlsolver has concrete type
    ldlsolver = kktsolver.ldlsolver
     return _kktsolver_update_inner!(kktsolver,ldlsolver,cones)
end


function _kktsolver_update_inner!(
    kktsolver::DirectLDLKKTSolver{T},
    ldlsolver::AbstractDirectLDLSolver{T},
    cones::CompositeCone{T}
) where {T}

    #real implementation is here, and now ldlsolver
    #will be compiled to something concrete.

    settings  = kktsolver.settings
    map       = kktsolver.map
    KKT       = kktsolver.KKT

    #Set the elements the W^tW blocks in the KKT matrix.
    get_Hs!(cones,kktsolver.Hsblocks)

    for (index, values) in zip(map.Hsblocks,kktsolver.Hsblocks)
        #change signs to get -W^TW
        # values .= -values
        @. values *= -one(T)
        _update_values!(ldlsolver,KKT,index,values)
    end

    #update the scaled u and v columns.
    cidx = 1        #which of the SOCs are we working on?

    for (i,K) = enumerate(cones)
        if isa(cones.cone_specs[i],SecondOrderConeT)

            η2 = K.η^2

            #off diagonal columns (or rows)
            _update_values!(ldlsolver,KKT,map.SOC_u[cidx],K.u)
            _update_values!(ldlsolver,KKT,map.SOC_v[cidx],K.v)
            _scale_values!(ldlsolver,KKT,map.SOC_u[cidx],-η2)
            _scale_values!(ldlsolver,KKT,map.SOC_v[cidx],-η2)


            #add η^2*(1/-1) to diagonal in the extended rows/cols
            _update_values!(ldlsolver,KKT,[map.SOC_D[cidx*2-1]],[-η2])
            _update_values!(ldlsolver,KKT,[map.SOC_D[cidx*2  ]],[+η2])

            cidx += 1
        end
    end

    return _kktsolver_regularize_and_refactor!(kktsolver, ldlsolver)

end

function _kktsolver_regularize_and_refactor!(
    kktsolver::DirectLDLKKTSolver{T},
    ldlsolver::AbstractDirectLDLSolver{T}
) where{T}

    settings      = kktsolver.settings
    map           = kktsolver.map
    KKT           = kktsolver.KKT
    Dsigns        = kktsolver.Dsigns
    diag_kkt      = kktsolver.work1
    diag_shifted  = kktsolver.work2

    if(settings.static_regularization_enable)

        # hold a copy of the true KKT diagonal
        @views diag_kkt .= KKT.nzval[map.diag_full]
        ϵ = _compute_regularizer(diag_kkt, settings)

        # compute an offset version, accounting for signs
        diag_shifted .= diag_kkt
        @inbounds for i in eachindex(Dsigns)
            if(Dsigns[i] == 1) diag_shifted[i] += ϵ
            else               diag_shifted[i] -= ϵ
            end
        end
        # overwrite the diagonal of KKT and within the ldlsolver
        _update_values!(ldlsolver,KKT,map.diag_full,diag_shifted)

        # remember the value we used.  Not needed,
        # but possibly useful for debugging
        kktsolver.diagonal_regularizer = ϵ

    end

    is_success = refactor!(ldlsolver,KKT)

    if(settings.static_regularization_enable)

        # put our internal copy of the KKT matrix back the way
        # it was. Not necessary to fix the ldlsolver copy because
        # this is only needed for our post-factorization IR scheme

        _update_values_KKT!(KKT,map.diag_full,diag_kkt)

    end

    return is_success
end


function _compute_regularizer(
    diag_kkt::AbstractVector{T},
    settings::Settings{T}
) where {T}

    maxdiag  = norm(diag_kkt,Inf);

    # Compute a new regularizer
    regularizer =  settings.static_regularization_constant +
                   settings.static_regularization_proportional * maxdiag

    return regularizer

end


function kktsolver_setrhs!(
    kktsolver::DirectLDLKKTSolver{T},
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
    kktsolver::DirectLDLKKTSolver{T},
    lhsx::Union{Nothing,AbstractVector{T}},
    lhsz::Union{Nothing,AbstractVector{T}}
) where {T}

    x = kktsolver.x
    (m,n,p) = (kktsolver.m,kktsolver.n,kktsolver.p)

    isnothing(lhsx) || (@views lhsx .= x[1:n])
    isnothing(lhsz) || (@views lhsz .= x[(n+1):(n+m)])

    return nothing
end


function kktsolver_solve!(
    kktsolver::DirectLDLKKTSolver{T},
    lhsx::Union{Nothing,AbstractVector{T}},
    lhsz::Union{Nothing,AbstractVector{T}}
) where {T}

    (x,b) = (kktsolver.x,kktsolver.b)
    solve!(kktsolver.ldlsolver,x,b)

    is_success = begin
        if(kktsolver.settings.iterative_refinement_enable)
            #IR reports success based on finite normed residual
            is_success = _iterative_refinement(kktsolver,kktsolver.ldlsolver)
        else
             # otherwise must directly verify finite values
            is_success = all(isfinite,x)
        end
    end

    if is_success
       kktsolver_getlhs!(kktsolver,lhsx,lhsz)
    end

    return is_success
end

function  _iterative_refinement(
    kktsolver::DirectLDLKKTSolver{T},
    ldlsolver::AbstractDirectLDLSolver{T}
) where{T}

    (x,b)   = (kktsolver.x,kktsolver.b)
    (e,dx)  = (kktsolver.work1, kktsolver.work2)
    settings = kktsolver.settings

    #iterative refinement params
    IR_reltol    = settings.iterative_refinement_reltol
    IR_abstol    = settings.iterative_refinement_abstol
    IR_maxiter   = settings.iterative_refinement_max_iter
    IR_stopratio = settings.iterative_refinement_stop_ratio

    KKTsym = kktsolver.KKTsym
    normb  = norm(b,Inf)

    #compute the initial error
    norme = _get_refine_error!(e,b,KKTsym,x)

    for i = 1:IR_maxiter

        # bail on numerical error
        if !isfinite(norme) return is_success = false end

        if(norme <= IR_abstol + IR_reltol*normb)
            # within tolerance, or failed.  Exit
            break
        end
        lastnorme = norme

        #make a refinement and continue
        solve!(ldlsolver,dx,e)

        #prospective solution is x + dx.   Use dx space to
        #hold it for a check before applying to x
        ξ = dx
        @. ξ += x
        norme = _get_refine_error!(e,b,KKTsym,ξ)

        improved_ratio = lastnorme/norme
        if(improved_ratio <  IR_stopratio)
            #insufficient improvement.  Exit
            if (improved_ratio > one(T))
                @. x = ξ #PJG: pointer swap might be faster
            end
            break
        end
        
        @. x = ξ  #PJG: pointer swap might be faster
    end

    #NB: "success" means only that we had a finite valued result
    return is_success = true
end


# computes e = b - Kξ, overwriting the first argument
# and returning its norm

function _get_refine_error!(
    e::AbstractVector{T},
    b::AbstractVector{T},
    KKTsym::Symmetric{T},
    ξ::AbstractVector{T}) where {T}

    @. e = b
    mul!(e,KKTsym,ξ,-1.,1.)   # e = b - Kξ

    return norm(e,Inf)

end
