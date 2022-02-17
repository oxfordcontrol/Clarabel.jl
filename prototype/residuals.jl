function residuals_update!(
    residuals::DefaultResiduals{T},
    variables::DefaultVariables{T},
    data::DefaultProblemData{T}
) where {T}

  #scalars used locally more than once
  qx        = dot(data.q,variables.x)
  bz        = dot(data.b,variables.z.vec)
  sz        = dot(variables.s.vec,variables.z.vec)
  xPx       = dot(variables.x, data.P, variables.x)

  #partial residual calc so we can check primal/dual
  #infeasibility conditions
  residuals.rx .= data.P * variables.x + data.A'* variables.z.vec
  residuals.rz .= data.A * variables.x + variables.s.vec

  #norms for infeasibility checks (scaled)
  residuals.norm_pinf  = norm(residuals.rx)
  residuals.norm_dinf  = norm(residuals.rz)

  #finish the residual calculation
  residuals.rx .= -residuals.rx - data.q * variables.τ
  residuals.rz .= +residuals.rz - data.b * variables.τ
  residuals.rτ  = qx + bz + variables.κ + xPx/variables.τ

  #relative residuals (scaled)
  residuals.norm_rz     = norm(residuals.rz)
  residuals.norm_rx     = norm(residuals.rx)

  #various dot products for later use (all with scaled variables)
  residuals.dot_qx  = qx
  residuals.dot_bz  = bz
  residuals.dot_sz  = sz
  residuals.dot_xPx = xPx

  return nothing
end
