
using ProgressMeter
using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using QuadGK
using Plots,NamedColors ; theme(:dark) #; plotlyjs()
using FileIO
using BenchmarkTools
using FFTW
using Random
Random.seed!(0)

push!(LOAD_PATH, abspath(@__DIR__,".."))

using  HawkesSimulator; const global H = HawkesSimulator

##

function mysim(γspikes::Real=1.5)
    # network parameters
    ne = 5
    τe = 10E-3
    he = 5.0 
    w_min = 1E-5
    w_max = 1/(ne-1) - 1E-5
    # set random weights
    w_mat = w_min.+(w_max-w_min).*rand(Float64, (ne,ne))
    # set zero diagonal
    w_mat[diagind(w_mat)] .= 0.0

    # expected neural rates (analytic) at the beginning, and in case of saturation.
    he_vec = fill(he,ne)
    ## create a population state and a trace
    # (the trace is necessary for the connection)
    pse,trae = H.population_state_exp_and_trace(ne,τe;label="population_e")
    connection_ee = H.ConnectionExpKernel(w_mat,trae)
    pop_e = H.PopulationExpKernel(pse,he_vec,(connection_ee,pse))
    n_spikes = Int(γspikes*1E7)
    the_network = H.RecurrentNetworkExpKernel((pop_e,))
    t_end = run_simulation!(the_network,n_spikes)
    return t_end
end



