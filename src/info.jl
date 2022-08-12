function info_update!(
    info::DefaultInfo{T},
    data::DefaultProblemData{T},
    variables::DefaultVariables{T},
    residuals::DefaultResiduals{T},
    settings::Settings{T},
    timers::TimerOutput
) where {T}

    #optimality termination check should be computed w.r.t
    #the pre-homogenization x and z variables.
    τinv = inv(variables.τ)

    #shortcuts for the equilibration matrices
    d = data.equilibration.d; dinv = data.equilibration.dinv
    e = data.equilibration.e; einv = data.equilibration.einv
    cscale = data.equilibration.c[]

    #primal and dual costs. dot products are invariant w.r.t
    #equilibration, but we still need to back out the overall
    #objective scaling term c
    xPx_τinvsq_over2 = residuals.dot_xPx * τinv * τinv / 2;
    info.cost_primal =  (+residuals.dot_qx*τinv + xPx_τinvsq_over2)/cscale
    info.cost_dual   =  (-residuals.dot_bz*τinv - xPx_τinvsq_over2)/cscale

    #primal and dual residuals.   Need to invert the equilibration
    info.res_primal  = scaled_norm(einv,residuals.rz) * τinv / (one(T) + data.normb)
    info.res_dual    = scaled_norm(dinv,residuals.rx) * τinv / (one(T) + data.normq)

    #primal and dual infeasibility residuals.   Need to invert the equilibration
    info.res_primal_inf = scaled_norm(dinv,residuals.rx_inf)
    info.res_dual_inf   = max(scaled_norm(dinv,residuals.Px),scaled_norm(einv,residuals.rz_inf))

    #absolute and relative gaps
    info.gap_abs   = residuals.dot_sz * τinv * τinv
    if(info.cost_primal > 0 && info.cost_dual < 0)
        info.gap_rel = 1/eps()
    else
        info.gap_rel = info.gap_abs / max(one(T),min(abs(info.cost_primal),abs(info.cost_dual)))
    end

    #κ/τ
    info.ktratio = variables.κ / variables.τ

    #solve time so far (includes setup!)
    info_get_solve_time!(info,timers)

end

function info_check_termination!(
    info::DefaultInfo{T},
    residuals::DefaultResiduals{T},
    settings::Settings{T},
    iter::Int
) where {T}

    #optimality
    #---------------------
    info.status = UNSOLVED  #ensure default state
    # println("current gap: ", min(info.gap_abs, info.gap_rel))

    if( ((info.gap_abs < settings.tol_gap_abs) || (info.gap_rel < settings.tol_gap_rel))
        && (info.res_primal < settings.tol_feas)
        && (info.res_dual   < settings.tol_feas)
    )
        info.status = SOLVED

    elseif info.ktratio > one(T)

        if (residuals.dot_bz < -settings.tol_infeas_rel) &&
            (info.res_primal_inf < -settings.tol_infeas_abs*residuals.dot_bz)

            info.status = PRIMAL_INFEASIBLE

        elseif (residuals.dot_qx < -settings.tol_infeas_rel) &&
                (info.res_dual_inf < -settings.tol_infeas_abs*residuals.dot_qx)

            info.status = DUAL_INFEASIBLE

        end
    end

    # YC: Terminate early when residuals diverge
    if iter > 0 && (info.res_dual > info.prev_res_dual || info.res_primal > info.prev_res_primal)
        # YC: small ktratio means the algorithm converges but feasibility residuals get stucked due to some numerical issues
        if info.ktratio < 1e-8 && (info.prev_gap_abs < settings.tol_gap_abs || info.prev_gap_rel < settings.tol_gap_rel)
            info.status = EARLY_TERMINATED
        end
        # YC: Severe numerical issue happens and we should stop it immediately
        if (info.res_dual > 100*info.prev_res_dual || info.res_primal > 100*info.prev_res_primal)
            info.status = EARLY_TERMINATED
        end
    end


    #time or iteration limits
    #----------------------
    if info.status == UNSOLVED

        if settings.max_iter  == info.iterations
            info.status = MAX_ITERATIONS

        elseif info.solve_time > settings.time_limit
            info.status = MAX_TIME

        end
    end

    #return TRUE if we settled on a final status
    return is_done = info.status != UNSOLVED
end

function info_save_prev_iterate(
    info::DefaultInfo{T},
    variables::DefaultVariables{T},
    prev_variables::DefaultVariables{T}
) where {T}
    info.prev_cost_primal    = info.cost_primal
    info.prev_cost_dual      = info.cost_dual
    info.prev_res_primal     = info.res_primal
    info.prev_res_dual       = info.res_dual
    info.prev_gap_abs        = info.gap_abs
    info.prev_gap_rel        = info.gap_rel

    prev_variables.x    .= variables.x
    prev_variables.s    .= variables.s
    prev_variables.z    .= variables.z
    prev_variables.τ     = variables.τ
    prev_variables.κ     = variables.κ
end

function info_reset_to_prev_iterates(
    info::DefaultInfo{T},
    variables::DefaultVariables{T},
    prev_variables::DefaultVariables{T}
) where {T}
    info.cost_primal    = info.prev_cost_primal
    info.cost_dual      = info.prev_cost_dual
    info.res_primal     = info.prev_res_primal
    info.res_dual       = info.prev_res_dual
    info.gap_abs        = info.prev_gap_abs
    info.gap_rel        = info.prev_gap_rel

    variables.x    .= prev_variables.x
    variables.s    .= prev_variables.s
    variables.z    .= prev_variables.z
    variables.τ     = prev_variables.τ
    variables.κ     = prev_variables.κ
end

function info_save_scalars(
    info::DefaultInfo{T},
    μ::T,
    α::T,
    σ::T,
    iter::Int
) where {T}

    info.μ = μ
    info.step_length = α
    info.sigma = σ
    info.iterations = iter

    return nothing
end


function info_reset!(
    info::DefaultInfo{T},
    timers::TimerOutput
) where {T}

    info.status     = UNSOLVED
    info.iterations = 0
    info.solve_time = 0

    #reset the solve! timer, but keep the setup!
    reset_timer!(timers["solve!"])

    return nothing
end


function info_get_solve_time!(
    info::DefaultInfo{T},
    timers::TimerOutput
) where {T}
    #TimerOutputs reports in nanoseconds
    info.solve_time = TimerOutputs.tottime(timers)*1e-9
    return nothing
end


function info_finalize!(
    info::DefaultInfo{T},
    timers::TimerOutput
) where {T}
    info_get_solve_time!(info,timers)
    return nothing
end
