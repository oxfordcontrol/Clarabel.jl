function residuals_update!(
    residuals::DefaultResiduals{T},
    variables::DefaultVariables{T},
    data::DefaultProblemData{T}
) where {T}

  # various inner products used multiple times
  qx  = dot(data.q,variables.x)
  bz  = dot(data.b,variables.z)
  sz  = dot(variables.s,variables.z)
  xPx = symdot(variables.x,data.Psym,variables.x)

  #partial residual calc so we can check primal/dual
  #infeasibility conditions

  #Same as:
  #residuals.rPx_inf .= -data.Psym * variables.x
  #residuals.rx_inf .= -data.A'* variables.z
  mul!(residuals.rPx_inf, data.Psym , variables.x, -1., 0.)
  mul!(residuals.rx_inf, data.A', variables.z, -1., 0.)

  #Same as:  residuals.rz_inf .=  data.A * variables.x + variables.s
  @. residuals.rz_inf = variables.s
  mul!(residuals.rz_inf, data.A, variables.x, 1., 1.)

  #complete the residuals
  @. residuals.rx = residuals.rPx_inf + residuals.rx_inf - data.q * variables.τ
  @. residuals.rz = residuals.rz_inf - data.b * variables.τ
  residuals.rτ    = qx + bz + variables.κ + xPx/variables.τ

  #save local versions
  residuals.dot_qx  = qx
  residuals.dot_bz  = bz
  residuals.dot_sz  = sz
  residuals.dot_xPx = xPx

  return nothing
end
