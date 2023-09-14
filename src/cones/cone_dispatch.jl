#-----------------------------------------------------
# macro for circumventing runtime dynamic dispatch
# on AbstractCones and trying to force a jumptable
# structure instead.   Must wrap a call to a function
# with an argument explicitly named "cone", and constructs
# a big if/else table testing the type of cone against
# the subtypes of AbstractCone
# -----------------------------------------------------

# macro dispatch won't work unless each cone type is completely specified, i.e. 
# we can't dispatch statically on the non-concrete types PowerCone{T} or 
# ExponentialCone{T}.  So we first need a way to expand each AbstractCone{T} 
# to its complete type, including the extra parameters in the exp / power cones 

# None if this would be necessary if StaticArrays could write to MArrays 
# with non-isbits types.  See here:  
# https://github.com/JuliaArrays/StaticArrays.jl/pull/749
# If the PR is accepted then the type dependent vector and matrix types 
# defined in CONE3D_M3T_TYPE and CONE3D_V3T_TYPE could be dropped, 
# and ExponentialCone  and PowerCone would no longer need hidden 
#internal parameters with  outer-only constructors.

# turns PowerCone{T} to PowerCone{T,M3T,V3T}
function _make_conetype_concrete(::Type{PowerCone},T::Type) 
    return PowerCone{T,CONE3D_M3T_TYPE(T),CONE3D_V3T_TYPE(T)}
end
# turns ExponentialCone{T} to ExponentialCone{T,M3T,V3T}
function _make_conetype_concrete(::Type{ExponentialCone},T::Type) 
    return ExponentialCone{T,CONE3D_M3T_TYPE(T),CONE3D_V3T_TYPE(T)}
end
# turns any other AbstractCone{T} to itself
_make_conetype_concrete(conetype,T::Type) = conetype{T}

function _conedispatch(x, call)

    # We do not set thetypes = subtypes(AbstractCone), but 
    # rather to entries in our dictionary of primitive cone
    # types.   This avoids adding CompositeCone itself to the
    # switchyard we construct here, but would also prevent 
    # the use of nested CompositeCones.  
    thetypes = collect(values(ConeDict))
    foldr((t, tail) -> :(if $x isa _make_conetype_concrete($t,T); $call else $tail end), thetypes, init=Expr(:block))
end

macro conedispatch(call)
    esc(_conedispatch(:cone, call))
end