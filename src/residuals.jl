function residuals_update!(
    residuals::DefaultResiduals{T},
    variables::DefaultVariables{T},
    data::DefaultProblemData{T}
) where {T}

  # various products used multiple times
  qx  = dot(data.q,variables.x)
  bz  = dot(data.b,variables.z)
  sz  = dot(variables.s,variables.z)
  mul!(residuals.Px,Symmetric(data.P),variables.x)
  xPx = dot(variables.x,residuals.Px)

  #partial residual calc so we can check primal/dual
  #infeasibility conditions

  #Same as:
  #residuals.rx_inf .= -data.A'* variables.z
  mul!(residuals.rx_inf, data.A', variables.z, -one(T), zero(T))

  #Same as:  residuals.rz_inf .=  data.A * variables.x + variables.s
  @. residuals.rz_inf = variables.s
  mul!(residuals.rz_inf, data.A, variables.x, one(T), one(T))

  #complete the residuals
  @. residuals.rx = residuals.rx_inf - residuals.Px - data.q * variables.τ
  @. residuals.rz = residuals.rz_inf - data.b * variables.τ
  residuals.rτ    = qx + bz + variables.κ + xPx/variables.τ

  #save local versions
  residuals.dot_qx  = qx
  residuals.dot_bz  = bz
  residuals.dot_sz  = sz
  residuals.dot_xPx = xPx

  return nothing
end
