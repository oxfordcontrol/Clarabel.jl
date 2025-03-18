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

     #Δz <- WΔz
    tmp .= step_z         
    mul_W!(K,:N,step_z,tmp)        

    #Δs <- W⁻¹Δs
    tmp .= step_s           
    mul_Winv!(K,:T,step_s,tmp)      

    #shift = W⁻¹Δs ∘ WΔz - σμe
    circ_op!(K,shift,step_s,step_z)         
    
    #cone will be self dual, so Primal/Dual not important
    scaled_unit_shift!(K,shift,-σμ,PrimalCone::PrimalOrDualCone)                         

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
    mul_W!(K,:T,out,work)

end