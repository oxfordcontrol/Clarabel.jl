using InteractiveUtils  #allows call to subtypes

# -----------------------------------------------------
# macro for circumventing runtime dynamic dispatch
# on AbstractCones and trying to force a jumptable
# structure instead.   Must wrap a call to a function
# with an argument explicitly named "cone", and constructs
# a big if/else table testing the type of cone against
# the subtypes of AbstractCone
# -----------------------------------------------------

function _conedispatch(type, x, call)
    thetypes = subtypes(getfield(@__MODULE__, type))
    foldr((t, tail) -> :(if $x isa $t; $call else $tail end), thetypes, init=Expr(:block))
end

macro conedispatch(call)
    esc(_conedispatch(:AbstractCone, :cone, call))
end

#for debugging.  Replace @conedispatch with @noop
#to disable the type expansion.
macro noop(call)
    esc(call)
end

# -----------------------------------------------------
# dispatch operators for multiple cones
# -----------------------------------------------------

function cones_all_symmetric(cones::ConeSet{T}) where {T}
    return any(is_symmetric, cones)
end

function cones_rectify_equilibration!(
    cones::ConeSet{T},
     δ::ConicVector{T},
     e::ConicVector{T}
) where{T}

    any_changed = false

    #we will update e <- δ .* e using return values
    #from this function.  default is to do nothing at all
    δ .= 1

    for (cone,δi,ei) in zip(cones,δ.views,e.views)
        @conedispatch any_changed |= rectify_equilibration!(cone,δi,ei)
    end

    return any_changed
end


function cones_update_scaling!(
    cones::ConeSet{T},
    s::ConicVector{T},
    z::ConicVector{T},
	μ::T
) where {T}

    # update cone scalings by passing subview to each of
    # the appropriate cone types.
    for (cone,si,zi) in zip(cones,s.views,z.views)
        @conedispatch update_scaling!(cone,si,zi,μ)
    end

    return nothing
end


function cones_set_identity_scaling!(
    cones::ConeSet{T}
) where {T}

    for cone in cones
        @conedispatch set_identity_scaling!(cone)
    end

    return nothing
end


# The WtW block for each cone.
function cones_get_WtW_blocks!(
    cones::ConeSet{T},
    WtWblocks::Vector{Vector{T}}
) where {T}

    for (cone, block) in zip(cones,WtWblocks)
        @conedispatch get_WtW_block!(cone,block)
    end
    return nothing
end

# YC:x = λ ∘ λ for symmetric cone and x = s for unsymmetric cones
function cones_affine_ds!(
    cones::ConeSet{T},
    x::ConicVector{T},
    s::ConicVector{T}
) where {T}

    for (cone,xi,si) in zip(cones,x.views,s.views)
        @conedispatch affine_ds!(cone,xi,si)
    end
    return nothing
end

# YC:   x = y ∘ z for symmetric cones
#       x = 3rd-correction for unsymmetric cones
# NB: could merge with 3rd-functions later
function cones_circ_op!(
    cones::ConeSet{T},
    x::ConicVector{T},
    y::ConicVector{T},
    z::ConicVector{T}
) where {T}

    for (cone,xi,yi,zi) in zip(cones,x.views,y.views,z.views)
        # don't implement it for unsymmetric cones
        if !(cone in NonsymmetricCones)
            @conedispatch circ_op!(cone,xi,yi,zi)
        end
    end
    return nothing
end

# x = λ \ z,  where λ is scaled internal
# variable for each cone
function cones_λ_inv_circ_op!(
    cones::ConeSet{T},
    x::ConicVector{T},
    z::ConicVector{T}
) where {T}

    for (cone,xi,zi) in zip(cones,x.views,z.views)
        # don't implement it for unsymmetric cones
        if !(cone in NonsymmetricCones)
            @conedispatch λ_inv_circ_op!(cone,xi,zi)
        end
    end
    return nothing
end

# x = y \ z
function cones_inv_circ_op!(
    cones::ConeSet{T},
    x::ConicVector{T},
    y::ConicVector{T},
    z::ConicVector{T}
) where {T}

    for cone in zip(cones,x.views,y.views,z.views)
        # don't implement it for unsymmetric cones
        if !(cone in NonsymmetricCones)
            @conedispatch inv_circ_op!(cone,xi,yi,zi)
        end
    end
    return nothing
end

# place a vector to some nearby point in the cone
# YC: only when there is no unsymmetric cone
function cones_shift_to_cone!(
    cones::ConeSet{T},
    z::ConicVector{T}
) where {T}

    for (cone,zi) in zip(cones,z.views)
        @conedispatch shift_to_cone!(cone,zi)
    end
    return nothing
end

# initialization when with unsymmetric cones
function unit_initialization!(
    cones::ConeSet{T},
    s::ConicVector{T},
    z::ConicVector{T}
) where {T}

    for (cone,si,zi) in zip(cones,s.views,z.views)
        @conedispatch unsymmetric_init!(cone,si,zi)
    end
    return nothing
end

# computes y = αWx + βy, or y = αWᵀx + βy, i.e.
# similar to the BLAS gemv interface.
#Warning: x must not alias y.
function cones_gemv_W!(
    cones::ConeSet{T},
    is_transpose::Symbol,
    x::ConicVector{T},
    y::ConicVector{T},
    α::T,
    β::T
) where {T}

    #@assert (x !== y)
    for (cone,xi,yi) in zip(cones,x.views,y.views)
        # don't implement it for unsymmetric cones
        if !(cone in NonsymmetricCones)
            @conedispatch gemv_W!(cone,is_transpose,xi,yi,α,β)
        end
    end
    return nothing
end

# compute ds in the combined step where λ ∘ (WΔz + W^{-⊤}Δs) = - ds
function cones_combined_ds!(
    cones::ConeSet{T},
    dz::ConicVector{T},
    ds::ConicVector{T},
    step_z::ConicVector{T},
    step_s::ConicVector{T},
    σμ::T
) where {T}

    for (cone,dzi,zi,si) in zip(cones,dz.views,step_z.views,step_s.views)
        #We compute the centering and the higher order correction parts in ds and save it in dz
        @conedispatch combined_ds!(cone,dzi,zi,si,σμ)
    end

    #We are relying on d.s = λ ◦ λ (symmetric) or d.s = s (unsymmetric) already from the affine step here
    ds .+= dz

    return nothing
end

# compute the generalized step Wᵀ(λ \ ds)
function cones_Wt_λ_inv_circ_ds!(
    cones::ConeSet{T},
    lz::ConicVector{T},
    rz::ConicVector{T},
    rs::ConicVector{T},
    Wtlinvds::ConicVector
) where {T}

    for (cone,lzi,rzi,rsi,Wtlinvdsi) in zip(cones,lz.views,rz.views,rs.views,Wtlinvds.views)
        @conedispatch Wt_λ_inv_circ_ds!(cone,lzi,rzi,rsi,Wtlinvdsi)
    end

    return nothing
end

# compute the generalized step of -WᵀWΔz
function cones_WtW_Δz!(
    cones::ConeSet{T},
    lz::ConicVector{T},
    ls::ConicVector{T},
    workz::ConicVector{T}
) where {T}

    for (cone,lzi,lsi,workzi) in zip(cones,lz.views,ls.views,workz.views)
        @conedispatch WtW_Δz!(cone,lzi,lsi,workzi)
    end

    return nothing
end

# maximum allowed step length over all cones
function cones_step_length(
    cones::ConeSet{T},
    dz::ConicVector{T},
    ds::ConicVector{T},
    dτ::T,
    dκ::T,
     z::ConicVector{T},
     s::ConicVector{T},
     τ::T,
     κ::T,
    α::T
) where {T}

    dz    = dz.views
    ds    = ds.views
    z     = z.views
    s     = s.views


    # YC: implement step search for symmetric cones first
    # NB: split the step search for symmetric and unsymmtric cones due to the complexity of the latter
    for (cone,type,dzi,dsi,zi,si) in zip(cones,cones.types,dz,ds,z,s)
        if (type in NonsymmetricCones)
            @conedispatch αzs = unsymmetric_step_length(cone,dzi,dsi,zi,si,α,cones.scaling)
            α = min(α,αzs)
        else
            @conedispatch (nextαz,nextαs) = step_length(cone,dzi,dsi,zi,si)
            α = min(α,nextαz,nextαs)
        end
    end

    return α
end

# check the distance to the boundary for unsymmetric cones
function check_μ_and_centrality(
    cones::ConeSet{T},
    step::DefaultVariables{T},
    variables::DefaultVariables{T},
    work::DefaultVariables{T},
    α::T,
    steptype::Symbol
) where {T}

    dz    = step.z
    ds    = step.s
    dτ    = step.τ
    dκ    = step.κ
    z     = variables.z
    s     = variables.s
    τ     = variables.τ
    κ     = variables.κ
    cur_z = work.z
    cur_s = work.s

    zs= dot(z,s)
    dzs = dot(dz,ds)
    s_dz = dot(s,dz)
    z_ds = dot(z,ds)

    central_coef = cones.degree + 1

    # YC: scaling parameter to avoid reaching the boundary of cones
        # when we compute barrier functions
    # NB: different choice of α yields different performance, don't know how to explain it,
    #       but we must need it. Otherwise, there would be numerical issues for barrier computation
    α *= T(0.995)

    length_exp = cones.type_counts[ExponentialConeT]
    ind_exp = cones.ind_exp
    length_pow = cones.type_counts[PowerConeT]
    ind_pow = cones.ind_pow
    scaling = cones.scaling
    η = cones.η

    for j = 1:50
        #Initialize μ
        μ = (zs + τ*κ + α*(s_dz + z_ds + dτ*κ + τ*dκ) + α^2*(dzs + dτ*dκ))/central_coef
        upper = cones.minDist*μ     #bound for boundary distance

        @. cur_z = z + α*dz
        @. cur_s = s + α*ds

        # #boundary check from ECOS and centrality check from Hypatia
        # # NB:   1) the update x+α*dx is inefficient right now and need to be rewritten later
        # #       2) symmetric cones use the central path as in CVXOPT
        # if boundary_check!(cur_z,cur_s,ind_exp,length_exp,upper) && boundary_check!(cur_z,cur_s,ind_pow,length_pow,upper) && check_centrality!(cones,cur_s,cur_z,μ,η)
        #     return α
        # else
        #     α *= scaling
        # end


        # ECOS: check centrality, functional proximity measure
        # NB: the update x+α*dx is inefficient right now and need to be rewritten later
        if !(boundary_check!(cur_z,cur_s,ind_exp,length_exp,upper) && boundary_check!(cur_z,cur_s,ind_pow,length_pow,upper))
            α *= scaling
            continue
        end
        barrier = central_coef*log(μ) - log(τ+α*dτ) - log(κ+α*dκ)

        for (cone,cur_si,cur_zi) in zip(cones,cur_s.views, cur_z.views)
            @conedispatch barrier += f_sum(cone, cur_si, cur_zi)
        end

        if barrier < 1.
            return α
        else
            α *= scaling    #backtrack line search
        end
        # println("centrality quite bad: ", barrier, " with ", central_coef)

    end

    if (steptype == :combined)
        error("get stalled with step size ", α)
    end

    return α
end

function boundary_check!(z,s,ind_cone,length_cone,upper)

    for i = 1:length_cone
        μi = dot(z.views[ind_cone[i]],s.views[ind_cone[i]])/3

        # ECOS: if too close to boundary
        if μi < upper
            println("var too close to boundary")
            return false
        end
    end

    return true
end

function check_centrality!(cones,s,z,μ,η)

    for (cone,si,zi) = zip(cones,s.views,z.views)
        @conedispatch _chk = _check_neighbourhood(cone,si,zi,μ,η)
        if !_chk
            return false
        end
    end

    return true
end
