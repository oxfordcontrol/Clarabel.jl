function check_termination!(
    status::DefaultStatus{T},
    data::DefaultProblemData{T},
    variables::DefaultVariables{T},
    residuals::DefaultResiduals{T},
    scalings::DefaultScalings{T},
    settings::Settings{T}
) where {T}

    #optimality termination check should be computed w.r.t
    #the unscaled x and z variables.
    τinv = 1 / variables.τ

    #primal and dual costs
    status.cost_primal =  residuals.dot_cx*τinv + residuals.dot_xPx * τinv * τinv / 2
    status.cost_dual   = -residuals.dot_bz*τinv - residuals.dot_xPx * τinv * τinv / 2

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
    if( (abs_gap < settings.tol_gap_abs) || (rel_gap < settings.tol_gap_rel)
        && (status.res_primal < settings.tol_feas)
        && (status.res_dual   < settings.tol_feas)
    )
        status.status = SOLVED

    #check for primal infeasibility
    #---------------------
    #PJG:Using unscaled variables here.   Double check normalization term (see notes)
    elseif(status.cost_dual > 0 &&
           residuals.norm_Atz/max(1,data.norm_c) < settings.tol_feas)
        status.status = PRIMAL_INFEASIBLE

    #check for dual infeasibility
    #---------------------
    #PJG:Using unscaled variables here.   Double check normalization term (see notes)
    elseif(status.cost_primal < 0 &&
           residuals.norm_Ax/max(1,data.norm_b) < settings.tol_feas)
        status.status = DUAL_INFEASIBLE
    end

    return is_done = status.status != UNSOLVED

end

function status_save_scalars(status,μ,α,σ,iter)

    status.gap = μ
    status.step_length = α
    status.sigma = σ
    status.iterations = iter

    return nothing
end


function status_reset!(status)

    status.status     = UNSOLVED
    status.iterations = 0
    status.solve_time = time()

    return nothing
end

function status_finalize!(status)

    status.solve_time = time() - status.solve_time

    return nothing
end
