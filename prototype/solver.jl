
# -------------------------------------
# setup!
# -------------------------------------
function Setup!(
    s::Solver{T},
    c::Vector{T},
    A::AbstractMatrix{T},
    b::Vector{T},
    cone_types::Vector{SupportedCones},
    cone_dims::Vector{Int},
    settings::Settings{T} = Settings()) where{T}

    cone_info   = ConeInfo(cone_types,cone_dims)

    s.settings  = settings
    s.data      = DefaultProblemData(c,A,b,cone_info)
    s.scalings  = DefaultConeScalings(cone_info)
    s.variables = DefaultVariables(s.data.n,s.data.m,cone_info)
    s.residuals = DefaultResiduals(s.data.n,s.data.m)
    s.kktsolver = DefaultKKTSolver(s.data)
    s.status    = DefaultStatus()

    # work variables for assembling step direction LHS/RHS
    s.step_rhs  = DefaultVariables(s.data.n,s.data.m,s.scalings.cone_info)
    s.step_lhs  = DefaultVariables(s.data.n,s.data.m,s.scalings.cone_info)

end


# -------------------------------------
# solve!
# -------------------------------------
function Solve!(
    s::Solver{T}) where{T}

    #various initializations
    ResetStatus(s.status)
    iter   = 0
    isdone = false

    #initial residuals and duality gap
    gap       = T(0)
    sigma     = T(0)

    #hi to user!
    PrintHeader(s)

    #initialize variables to some reasonable starting point
    DefaultStart!(s)

    #----------
    # main loop
    #----------
    while iter <= s.settings.max_iter

        iter += 1

        #update the residuals
        #--------------
        UpdateResiduals!(s)

        #calculate duality gap (scaled)
        #--------------
        μ = CalcMu(s.variables, s.residuals, s.scalings)

        #convergence check and printing
        #--------------
        isdone = CheckTermination(s.status,s.data,s.variables,s.residuals,s.scalings,s.settings)
        PrintStatus(s.status,s.settings)

        isdone && break

        #update the scalings
        #--------------
        UpdateScalings!(s.scalings,s.variables)

        #update the KKT system
        #--------------
        UpdateKKTSystem!(s.kktsolver,s.scalings)

        #calculate KKT solution for constant terms
        #--------------
        SolveKKTConstantRHS!(s.kktsolver,s.data)

        #calculate the affine step
        #--------------
        CalcAffineStepRHS!(s.step_rhs, s.residuals, s.variables, s.scalings)
        SolveKKT!(s.kktsolver, s.step_lhs, s.step_rhs, s.variables, s.scalings, s.data)

        #calculate step length and centering parameter
        #--------------
        α = CalcStepLength(s.variables,s.step_lhs,s.scalings)
        σ = CalcCenteringParameter(α)

        #calculate the combined step and length
        #--------------
        CalcCombinedStepRHS!(s.step_rhs,s.residuals,s.variables,s.scalings,s.step_lhs,σ,μ)
        SolveKKT!(s.kktsolver, s.step_lhs, s.step_rhs, s.variables, s.scalings, s.data)

        #compute final step length and update the current iterate
        #--------------
        α = 0.99*CalcStepLength(s.variables,s.step_lhs,s.scalings) #PJG: make tunable
        AddToVariables!(s.variables,s.step_lhs,α)

        #record scalar values from this iteration
        SaveScalarStatus(s.status,μ,α,σ,iter)

    end

    FinalizeStatus(s.status)
    PrintFooter(s.status,s.settings)

end

function ResetStatus(status)

    status.status     = UNSOLVED
    status.iterations = 0
    status.solve_time = time()

end

function FinalizeStatus(status)

    status.solve_time = time() - status.solve_time

end

function SaveScalarStatus(status,μ,α,σ,iter)

    status.gap = μ
    status.step_length = α
    status.sigma = σ
    status.iterations = iter

end


function AddToVariables!(
    variables::DefaultVariables{T},
    step::DefaultVariables{T}, α::T) where {T}

    variables.x     .+= α*step.x
    variables.s.vec .+= α*step.s.vec
    variables.z.vec .+= α*step.z.vec
    variables.τ      += α*step.τ
    variables.κ      += α*step.κ

end



function CalcStepLength(
    variables::DefaultVariables{T},
    step::DefaultVariables{T},
    scalings::DefaultConeScalings{T},) where {T}

    ατ    = step.τ < 0 ? -variables.τ / step.τ : 1/eps(T)
    ακ    = step.κ < 0 ? -variables.κ / step.κ : 1/eps(T)
    αcone = step_length(scalings, step.z, step.s, variables.z, variables.s, variables.λ )

    α     = min(ατ,ακ,αcone,1.)

end

# Mehrotra heuristic
function CalcCenteringParameter(α::T) where{T}

    σ = (1-α)^3

end


function CalcAffineStepRHS!(
    d::DefaultVariables{T},
    r::DefaultResiduals{T},
    variables::DefaultVariables{T},
    scalings::DefaultConeScalings{T}) where{T}

    d.x     .=  r.rx
    d.z.vec .= -r.rz .+ variables.s.vec
    circle_op!(scalings, d.s, variables.λ, variables.λ)
    d.τ      =  r.rτ
    d.κ      =  variables.τ * variables.κ

end

# PJG: CalcCombinedStepRHS! modifies the step values
# in place to be economical with memory
function CalcCombinedStepRHS!(
    d::DefaultVariables{T},
    r::DefaultResiduals{T},
    variables::DefaultVariables{T},
    scalings::DefaultConeScalings{T},
    step::DefaultVariables{T},
    σ::T, μ::T) where {T}

    # assumes that the affine RHS currently occupies d,
    # so only applies incremental changes to get the
    # combined corrector step
    d.x .*= (1. - σ)
    d.τ  *= (1. - σ)
    d.κ  += step.τ * step.κ - σ*μ

    # d.s and d.z are  harder if we want to be
    # economical with allocated memory.  Modify the
    # step.z and step.s vectors in place since they
    # are from the affine step and not needed now.
    # Also use d.z as temporary space to hold
    # W⁻¹Δs ∘ WΔz
    gemv_W!(scalings,  false, step.z, step.z,  1., 0.)        #Δz = WΔz
    gemv_Winv!(scalings, false, step.s, step.s,  1., 0.)      #Δs = W⁻¹Δs
    circle_op!(scalings, d.z, step.s, step.z)                 #tmp = W⁻¹Δs ∘ WΔz
    add_scaled_e!(scalings,d.z,-σ*μ)                          #tmp = tmp -σμe
    d.s.vec .+= d.z.vec

    # now build d.z from scratch
    inv_circle_op!(scalings, d.z, variables.λ, d.s)           #dz = λ \ ds
    gemv_W!(scalings, false, d.z, d.z, 1., 0.)                #dz = Wdz
    d.z.vec .+= -(1-σ).*r.rz

end

function DefaultStart!(s::Solver{T}) where {T}

    #set all scalings to identity (or zero for the zero cone)
    IdentityScalings!(s.scalings,s.variables)
    #Refactor
    UpdateKKTSystem!(s.kktsolver,s.scalings)
    #solve for primal/dual initial points via KKT
    SolveKKTInitialPoint!(s.kktsolver,s.variables,s.data)
    #fix up (z,s) so that they are in the cone
    shift_to_cone!(s.scalings, s.variables.s)
    shift_to_cone!(s.scalings, s.variables.z)

    s.variables.τ = 1
    s.variables.κ = 1

end


function CalcMu(
    variables::DefaultVariables{T},
    residuals::DefaultResiduals{T},
    scalings::DefaultConeScalings{T}) where {T}

  μ = (residuals.dot_sz + variables.τ * variables.κ)/(scalings.total_order + 1)

end


function UpdateResiduals!(
    s::Solver{T}) where {T}

  residuals = s.residuals
  data      = s.data
  variables = s.variables

  #scalars used locally more than once
  cx        = dot(data.c,variables.x)
  bz        = dot(data.b,variables.z.vec)
  sz        = dot(variables.s.vec,variables.z.vec)

  #partial residual calc so I can catch the
  #norms of the matrix vector products
  residuals.rx = data.A'* variables.z.vec
  residuals.rz = data.A * variables.x

  #matrix vector product norm (scaled)
  residuals.norm_Ax  = norm(residuals.rz)
  residuals.norm_Atz = norm(residuals.rx)

  #finish the residual calculation
  residuals.rx .= -residuals.rx - data.c * variables.τ
  residuals.rz .= +residuals.rz - data.b * variables.τ + variables.s.vec
  residuals.rτ = cx + bz + variables.κ

  #relative residuals (scaled)
  residuals.norm_rz     = norm(residuals.rz)
  residuals.norm_rx     = norm(residuals.rx)

  #various dot products for later use (all with scaled variables)
  residuals.dot_cx = cx
  residuals.dot_bz = bz
  residuals.dot_sz = sz

end

function CheckTermination(
    status::DefaultStatus{T},
    data::DefaultProblemData{T},
    variables::DefaultVariables{T},
    residuals::DefaultResiduals{T},
    scalings::DefaultConeScalings{T},
    settings::Settings{T}) where {T}

    #optimality termination check should be computed w.r.t
    #the unscaled x and z variables.
    τinv = 1 / variables.τ

    #primal and dual costs
    status.cost_primal =  residuals.dot_cx*τinv
    status.cost_dual   = -residuals.dot_bz*τinv

    #primal and dual residuals
    status.res_primal  = norm(residuals.rx) * τinv / max(1,data.norm_c)
    status.res_dual    = norm(residuals.rz) * τinv / max(1,data.norm_b)

    #absolute and relative gaps
    abs_gap   = residuals.dot_sz * τinv * τinv
    if(status.cost_primal > 0 && status.cost_dual < 0)
        rel_gap = 1/eps()
    else
        rel_gap = abs_gap / min(abs(status.cost_primal),abs(status.cost_dual))
    end

    #κ/τ
    status.ktratio = variables.κ / variables.τ

    #check for convergence
    #---------------------
    if(    (abs_gap < settings.tol_gap_abs) || (rel_gap < settings.tol_gap_rel)
        && ( status.res_primal < settings.tol_feas)
        && ( status.res_dual   < settings.tol_feas)
      )
        status.status = SOLVED

    #check for primal infeasibility
    #---------------------
    #PJG:Using unscaled variables here.   Double check normalization term (see notes)
    elseif(status.cost_dual > 0 && residuals.norm_Atz / max(1,data.norm_c) < settings.tol_feas)
        status.status = PRIMAL_INFEASIBLE

    #check for dual infeasibility
    #---------------------
    #PJG:Using unscaled variables here.   Double check normalization term (see notes)
    elseif(status.cost_primal < 0 && residuals.norm_Ax / max(1,data.norm_b) < settings.tol_feas)
        status.status = DUAL_INFEASIBLE
    end

    return is_done = status.status != UNSOLVED

end
