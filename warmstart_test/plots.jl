using Plots
using Statistics
function plot_warmstart(warm_start,cold_start,filename)
    @assert(length(warm_start) == length(cold_start))
    # Create the indices
    indices = 1:length(warm_start)

    # Plot the vectors
    scatter(indices, warm_start, marker=:circle, label="warm_start")
    scatter!(indices, cold_start, marker=:diamond, label="cold_start")

    xlabel!("Times")
    ylabel!("Iterations")

    ymax = Int(ceil(max(maximum(warm_start),maximum(cold_start))/5.0)*5)
    ylims!(0, ymax)

    # Save the plot
    savefig(filename*".pdf")
end

function plot_varying_warmstart(warm_start_b,warm_start_q,warm_start_A,class)
    @assert(length(warm_start_b) == length(warm_start_q) && length(warm_start_A) == length(warm_start_q))
    # Create the indices
    indices = 10.0.^(range(-3,0,length=length(warm_start_b)))

    # Plot the vectors
    scatter(indices, warm_start_b, xscale=:log10, marker=:circle, label="b")
    scatter!(indices, warm_start_q, xscale=:log10, marker=:circle, label="q")
    scatter!(indices, warm_start_A, xscale=:log10, marker=:cross, label="A")

    xlabel!("Perturbation δ")
    ylabel!("Geometric mean")
    ylims!(0, 1.1)
    xlims!(10^(-3),10^0)

    # Save the plot
    savefig("./results//"*class*"_varying_disturbance.pdf")
end