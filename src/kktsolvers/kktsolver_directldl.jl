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
    work_e::Vector{T}
    work_dx::Vector{T}

    #KKT mapping from problem data to KKT
    map::LDLDataMap

    #the expected signs of D in KKT = LDL^T
    Dsigns::Vector{Int}

    # a vector for storing the WtW blocks
    # on the in the KKT matrix block diagonal
    WtWblocks::Vector{Vector{T}}

    #unpermuted KKT matrix
    KKT::SparseMatrixCSC{T,Int}

    #symmetric view for residual calcs
    KKTsym::Symmetric{T, SparseMatrixCSC{T,Int}}

    #settings just points back to the main solver settings.
    #Required since there is no separate LDL settings container
    settings::Settings{T}

    #the direct linear LDL solver
    ldlsolver::AbstractDirectLDLSolver{T}

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
        Dsigns = ones(Int,n+m+p)
        _fill_Dsigns!(Dsigns,m,n,p)

        #updates to the diagonal of KKT will be
        #assigned here before updating matrix entries
        WtWblocks = _allocate_kkt_WtW_blocks(T, cones)

        #which LDL solver should I use?
        ldlsolverT = _get_ldlsolver_type(settings.direct_solve_method)

        #does it want a :triu or :tril KKT matrix?
        kktshape = required_matrix_shape(ldlsolverT)
        KKT, map = _assemble_kkt_matrix(P,A,cones,kktshape)

        #KKT will be triu data only, but we will want
        #the following to allow products like KKT*x
        KKTsym = Symmetric(KKT)

        #the LDL linear solver engine
        ldlsolver = ldlsolverT{T}(KKT,Dsigns,settings)

        if(settings.static_regularization_enable)
            ϵ = settings.static_regularization_eps
            offset_values!(ldlsolver,map.diagP,ϵ)
        end

        return new(m,n,p,x,b,work_e,work_dx,map,Dsigns,WtWblocks,KKT,KKTsym,settings,ldlsolver)
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

    #the expected signs of D in LDL
    Dsigns[n+1:n+m] .= -1

    #the trailing block of p entries should
    #have alternating signs
    sign = -1
    for idx = (n+m+1):(n+m+p)
        Dsigns[idx] = sign
        sign *= -1
    end
end

#update entries in the KKT matrix using the
#given index into its CSC representation
function _update_values!(
    kktsolver::DirectLDLKKTSolver{T},
    index::Vector{Ti},
    values::Vector{T}
) where{T,Ti}

    #Update values in the KKT matrix K
    kktsolver.KKT.nzval[index] .= values

    #give the LDL subsolver an opportunity to update the same
    #values if needed.   This latter is useful for QDLDL
    #since it stores its own permuted copy
    update_values!(kktsolver.ldlsolver,index,values)



end

#offset entries in the KKT matrix using the
#given index into its CSC representation and
#an optional vector of signs
function _offset_values!(
    kktsolver::DirectLDLKKTSolver{T},
    index::Vector{Int},
    offset::Union{T,Vector{T}},
    signs::Union{Int,Vector{Int}} = 1
) where{T}

    #Update values in the KKT matrix K
    @. kktsolver.KKT.nzval[index] += offset*signs

    #give the LDL subsolver an opportunity to update the same
    #values if needed.   This latter is useful for QDLDL
    #since it stores its own permuted copy
    offset_values!(kktsolver.ldlsolver, index, offset, signs)

end

function kktsolver_update!(
    kktsolver::DirectLDLKKTSolver{T},
    cones::ConeSet{T}
) where {T}

    (m,n,p) = (kktsolver.m,kktsolver.n,kktsolver.p)

    settings  = kktsolver.settings
    ldlsolver = kktsolver.ldlsolver
    map       = kktsolver.map


    #Set the elements the W^tW blocks in the KKT matrix.
    cones_get_WtW_blocks!(cones,kktsolver.WtWblocks)
    for (index, values) in zip(map.WtWblocks,kktsolver.WtWblocks)
        #change signs to get -W^TW
        values .= -values
        _update_values!(kktsolver,index,values)
    end

    #update the scaled u and v columns.
    cidx = 1        #which of the SOCs are we working on?

    for (i,K) = enumerate(cones)
        if(cones.types[i] == SecondOrderConeT)

                η2 = K.η^2

                #off diagonal columns (or rows)
                _update_values!(kktsolver,map.SOC_u[cidx],(-η2).*K.u)
                _update_values!(kktsolver,map.SOC_v[cidx],(-η2).*K.v)

                #add η^2*(1/-1) to diagonal in the extended rows/cols
                _update_values!(kktsolver,[map.SOC_D[cidx*2-1]],[-η2])
                _update_values!(kktsolver,[map.SOC_D[cidx*2  ]],[+η2])

                cidx += 1
        end

    end

    #Perturb the diagonal terms WtW that we have just overwritten
    #with static regularizers.  Note that we don't want to shift
    #elements in the ULHS #(corresponding to P) since we already
    #shifted them at initialization and haven't overwritten it
    if(settings.static_regularization_enable)
        ϵ = settings.static_regularization_eps
        _offset_values!(kktsolver,map.diag_full,ϵ,kktsolver.Dsigns)
        _offset_values!(kktsolver,map.diagP,-ϵ)  #undo to the P shift
    end

    #refactor with new data
    refactor!(ldlsolver,kktsolver.KKT)

    return nothing
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

    isnothing(lhsx) || (lhsx .= x[1:n])
    isnothing(lhsz) || (lhsz .= x[(n+1):(n+m)])

    return nothing
end


function kktsolver_solve!(
    kktsolver::DirectLDLKKTSolver{T},
    lhsx::Union{Nothing,AbstractVector{T}},
    lhsz::Union{Nothing,AbstractVector{T}}
) where {T}

    (x,b) = (kktsolver.x,kktsolver.b)
    solve!(kktsolver.ldlsolver,x,b)

    if(kktsolver.settings.iterative_refinement_enable)
        iterative_refinement(kktsolver,x,b)
    end


    kktsolver_getlhs!(kktsolver,lhsx,lhsz)

    return nothing
end

function iterative_refinement(kktsolver::DirectLDLKKTSolver{T},x,b) where{T}

    e  = kktsolver.work_e
    dx = kktsolver.work_dx
    settings = kktsolver.settings

    #iterative refinement params
    IR_reltol    = settings.iterative_refinement_reltol
    IR_abstol    = settings.iterative_refinement_abstol
    IR_maxiter   = settings.iterative_refinement_max_iter
    IR_stopratio = settings.iterative_refinement_stop_ratio

    #Note that K is only triu data, so need to
    #be careful when computing the residual here
    K      = kktsolver.KKT
    KKTsym = kktsolver.KKTsym
    lastnorme = Inf

    normb = norm(b,Inf)

    for i = 1:IR_maxiter

        #this error = b - (K+ϵD)ξ + ϵDξ, where
        #the ϵD term is from the static regularizer
        e .= b
        mul!(e,KKTsym,x,-1.,1.)   #  b - Kξ

        if(settings.static_regularization_enable)
            ϵ = settings.static_regularization_eps
            @. e += ϵ * kktsolver.Dsigns
        end

        norme = norm(e,Inf)

        # test for convergence before committing
        # to a refinement step
        if(norme <= IR_abstol + IR_reltol*normb)
            break
        end

        #Halt if we haven't improved by at least the
        #halting ratio since the last pass through
        if(lastnorme/norme < IR_stopratio)
            break
        end

        #make a refinement and continue
        solve!(kktsolver.ldlsolver,dx,e)
        x .+= dx
    end

    return nothing
end
