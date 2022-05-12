# -------------------------------------
# KKTSolver using direct LDL factorisation
# -------------------------------------

mutable struct DirectLDLKKTSolver{T} <: AbstractKKTSolver{T}

    # problem dimensions
    m::Int; n::Int; p::Int

    # Left and right hand sides for solves
    x::Vector{T}
    b::Vector{T}

    #KKT mapping from problem data to KKT
    map::LDLDataMap

    #the expected signs of D in KKT = LDL^T
    Dsigns::Vector{Int}

    # a vector for storing the WtW blocks
    # on the in the KKT matrix block diagonal
    WtWblocks::Vector{Vector{T}}

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

        #the expected signs of D in LDL
        Dsigns = Vector{Int}(undef,n+m+p)
        _fill_Dsigns!(Dsigns,m,n,p)

        #updates to the diagonal of KKT will be
        #assigned here before updating matrix entries
        WtWblocks = _allocate_kkt_WtW_blocks(T, cones)

        #which LDL solver should I use?
        ldlsolverT = _get_ldlsolver_type(settings.direct_solve_method)

        #does it want a :triu or :tril KKT matrix?
        kktshape = required_matrix_shape(ldlsolverT)
        KKT, map = _assemble_kkt_matrix(P,A,cones,kktshape)

        #the LDL linear solver engine
        ldlsolver = ldlsolverT{T}(KKT,Dsigns,settings)

        if(settings.static_regularization_enable)
            ϵ = settings.static_regularization_eps
            offset_values!(ldlsolver,map.diagP,ϵ)
        end

        return new(m,n,p,x,b,map,Dsigns,WtWblocks,settings,ldlsolver)
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


function kktsolver_update!(
    kktsolver::DirectLDLKKTSolver{T},
    cones::ConeSet{T}
) where {T}

    settings  = kktsolver.settings
    ldlsolver = kktsolver.ldlsolver
    map       = kktsolver.map


    #Set the elements the W^tW blocks in the KKT matrix.
    cones_get_WtW_blocks!(cones,kktsolver.WtWblocks)
    for (index, values) in zip(map.WtWblocks,kktsolver.WtWblocks)
        #change signs to get -W^TW
        values .= -values
        update_values!(ldlsolver,index,values)
    end

    #update the scaled u and v columns.
    cidx = 1        #which of the SOCs are we working on?

    for (i,K) = enumerate(cones)
        if(cones.types[i] == SecondOrderConeT)

                η2 = K.η^2

                #off diagonal columns (or rows)
                update_values!(ldlsolver,map.SOC_u[cidx],K.u)
                update_values!(ldlsolver,map.SOC_v[cidx],K.v)
                scale_values!(ldlsolver,map.SOC_u[cidx],-η2)
                scale_values!(ldlsolver,map.SOC_v[cidx],-η2)

                #add η^2*(1/-1) to diagonal in the extended rows/cols
                update_values!(ldlsolver,[map.SOC_D[cidx*2-1]],[-η2])
                update_values!(ldlsolver,[map.SOC_D[cidx*2  ]],[+η2])

                cidx += 1
        end

    end

    #Perturb the diagonal terms WtW that we have just overwritten
    #with static regularizers.  Note that we don't want to shift
    #elements in the ULHS #(corresponding to P) since we already
    #shifted them at initialization and haven't overwritten it
    if(settings.static_regularization_enable)
        ϵ = settings.static_regularization_eps
        offset_values!(ldlsolver,map.diag_full,ϵ,kktsolver.Dsigns)
        offset_values!(ldlsolver,map.diagP,-ϵ)  #undo to the P shift
    end

    #refactor with new data
    refactor!(ldlsolver)

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
    solve!(kktsolver.ldlsolver,x,b,kktsolver.settings)
    kktsolver_getlhs!(kktsolver,lhsx,lhsz)

    return nothing
end
