# find the maximum step length α≥0 so that
# q + α*dq stays in an exponential or power
# cone, or their respective dual cones.
#
# NB: Not for use as a general checking
# function because cone lengths are hardcoded
# to R^3 for faster execution.  

# PJG: I reintroduced 'work' here since only the caller 
# will know how to make an appropriately typed SizedArray
# or MArray.   I guess this way we only end up the true 
# step returned in the first argument

function _step_length_3d_cone(
    wq::AbstractVector{T},
    dq::AbstractVector{T},
    q::AbstractVector{T},
    α_init::T,
    α_min::T,
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
        if (α *= backtrack) < α_min
            α = zero(T)
            break
        end
    end
    return α
end


# shift = W⁻¹Δs ∘ WΔz - σμe
@inline function _combined_ds_shift_symmetric!(
    K::Union{NonnegativeCone{T},SecondOrderCone{T},PSDTriangleCone{T}},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    σμ::T
) where {T}

    # The shift must be assembled carefully if we want to be economical with
    # allocated memory.  Will modify the step.z and step.s in place since
    # they are from the affine step and not needed anymore.
    #
    # We can't have aliasing vector arguments to gemv_W or gemv_Winv, so 
    # we need a temporary variable to assign #Δz <= WΔz and Δs <= W⁻¹Δs

    #shift vector used as workspace for a few steps 
    tmp = shift              

     #Δz <- Wdz
    tmp .= step_z         
    mul_W!(K,:N,step_z,tmp,one(T),zero(T))        

    #Δs <- W⁻¹Δs
    tmp .= step_s           
    mul_Winv!(K,:T,step_s,tmp,one(T),zero(T))      

    #shift = W⁻¹Δs ∘ WΔz - σμe
    circ_op!(K,shift,step_s,step_z)                 
    add_scaled_e!(K,shift,-σμ)                       

    return nothing
end


# out = Wᵀ(λ \ ds)
@inline function _Δs_from_Δz_offset_symmetric!(
    K::Union{NonnegativeCone{T},SecondOrderCone{T},PSDTriangleCone{T}},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T}
) where {T}

    #tmp = λ \ ds 
    λ_inv_circ_op!(K,work,ds) 

    #out = Wᵀ(λ \ ds) = Wᵀ(work) 
    mul_W!(K,:T,out,work,one(T),zero(T))

end