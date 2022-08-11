

# find the maximum step length α≥0 so that
# q + α*dq stays in an exponential or power
# cone, or their respective dual cones.
#
# NB: Not for use as a general checking
# function because cone lengths are hardcoded
# to R^3 for faster execution


function _step_length_powcone_or_expcone(
    wq::AbstractVector{T},
    dq::AbstractVector{T},
    q::AbstractVector{T},
    α_init::T,
    backtrack::T,
    is_in_cone_fcn::Function
) where {T}

    α = α_init

    while true

        #@. wq = q + α*dq
        @inbounds for i = 1:3
            wq[i] = q[i] + α*dq[i]
        end

        if is_in_cone_fcn(wq)
            break
        end

        if (α *= backtrack) < 1e-4
            α = zero(T)
            break
        end

    end

    return α
end
