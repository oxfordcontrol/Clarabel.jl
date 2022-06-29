function info_update!(
    info::DefaultInfo{T},
    data::DefaultProblemData{T},
    variables::DefaultVariables{T},
    residuals::DefaultResiduals{T},
    equil::DefaultEquilibration{T},
    settings::Settings{T}
) where {T}

    #optimality termination check should be computed w.r.t
    #the pre-homogenization x and z variables.
    τinv = inv(variables.τ)

    #shortcuts for the equilibration matrices
    D = equil.D; Dinv = equil.Dinv
    E = equil.E; Einv = equil.Einv
    cscale = equil.c[]

    #primal and dual costs. dot products are invariant w.r.t
    #equilibration, but we still need to back out the overall
    #objective scaling term c
    info.cost_primal =  (+residuals.dot_qx*τinv + residuals.dot_xPx * τinv * τinv / 2)/cscale
    info.cost_dual   =  (-residuals.dot_bz*τinv - residuals.dot_xPx * τinv * τinv / 2)/cscale

    #primal and dual residuals.   Need to invert the equilibration
    info.res_primal  = scaled_norm(Einv,residuals.rz) * τinv
    info.res_dual    = scaled_norm(Dinv,residuals.rx) * τinv

    #primal and dual infeasibility residuals.   Need to invert the equilibration
    info.res_primal_inf = scaled_norm(Dinv,residuals.rx_inf)
    info.res_dual_inf   = max(scaled_norm(Dinv,residuals.Px),scaled_norm(Einv,residuals.rz_inf))

    #absolute and relative gaps
    info.gap_abs   = residuals.dot_sz * τinv * τinv
    if(info.cost_primal > 0 && info.cost_dual < 0)
        info.gap_rel = 1/eps()
    else
        info.gap_rel = info.gap_abs / min(abs(info.cost_primal),abs(info.cost_dual))
    end

    #κ/τ
    info.ktratio = variables.κ / variables.τ

    #solve time so far (includes setup!)
    info_get_solve_time!(info)

end

function info_check_termination!(
    info::DefaultInfo{T},
    residuals::DefaultResiduals{T},
    settings::Settings{T}
) where {T}

    #optimality
    #---------------------
    info.status = UNSOLVED  #ensure default state
    println("current gap: ", min(info.gap_abs, info.gap_rel))
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

    #time or iteration limits
    #----------------------
    if info.status == UNSOLVED

        if settings.max_iter  == info.iterations
            info.status = MAX_ITERATIONS

        elseif settings.time_limit > zero(T) && info.solve_time > settings.time_limit
            info.status = MAX_TIME

        end
    end

    #return TRUE if we settled on a final status
    return is_done = info.status != UNSOLVED
end


function info_save_scalars(info,μ,α,σ,iter)

    info.μ = μ
    info.step_length = α
    info.sigma = σ
    info.iterations = iter

    return nothing
end


function info_reset!(info)

    info.status     = UNSOLVED
    info.iterations = 0
    info.solve_time = 0

    #reset the solve! timer, but keep the setup!
    reset_timer!(info.timer["solve!"])

    return nothing
end


function info_get_solve_time!(info)

    #TimerOutputs reports in nanoseconds
    info.solve_time = TimerOutputs.tottime(info.timer)*1e-9
    return nothing
end


function info_finalize!(info)

    info_get_solve_time!(info)
    return nothing
end
