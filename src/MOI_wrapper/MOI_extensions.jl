using MathOptInterface 
using ..Clarabel
const DefaultInt = Clarabel.DefaultInt

# This structure is parallel to the our API type, but 
# it is subtyped from MOI.AbstractVectorSet rather than 
# our own user API SupportedCone type.   

#NB: This "GenPowerCone" will end up within Clarabel.MOI.  
# It is not the same type as the one our top level API.

struct GenPowerCone{T<:Real} <: MathOptInterface.AbstractVectorSet
    α::Vector{T}
    dim2::DefaultInt
    function GenPowerCone(α::AbstractVector{T}, dim2::DefaultInt) where{T}
        @assert all(α .> zero(T))
        @assert isapprox(sum(α),one(T), atol=eps()*length(α)/2)
        new{T}(collect(α), dim2)
    end
end 

GenPowerCone(args...) = GenPowerCone{Float64}(args...)

# enable use of this type as a MOI constraint type
MathOptInterface.dimension(cone::GenPowerCone) = length(cone.α) + cone.dim2
Base.copy(cone::GenPowerCone{Float64}) = deepcopy(cone)