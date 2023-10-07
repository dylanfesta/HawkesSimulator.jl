
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


# this function runs the network dynamics
# it simply runs one-spike iteration steps for the given number of spikes
# returns the time at the end of the simulation
function run_simulation!(network,num_spikes;t_start::Float64=0.0)
  t_now = t_start
  # H.reset!(network) # clear spike trains etc
  for _ in 1:num_spikes
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end

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
    # add recorders
    Δt_wrec = 2.0
    Tstart_rec = 1*3600.0
    Tend_rec =  2*3600.0
    rec_spikes_e = H.RecSomeTrain(Int64(1E7),[1,2,3,4]; population_rec=1,
        Tstart = Tstart_rec,Tend = Tend_rec)
    rec_wee = H.RecTheseWeights(connection_ee.weights,Δt_wrec,Tend_rec;
        Tstart=Tstart_rec)
    the_network = H.RecurrentNetworkExpKernel((pop_e,),(rec_spikes_e,rec_wee))
    # the_network = H.RecurrentNetworkExpKernel((pop_e,),(rec_wee,))
    # the_network = H.RecurrentNetworkExpKernel((pop_e,),(rec_spikes_e,))
    t_end = run_simulation!(the_network,n_spikes)
    rec_w_content = H.get_content(rec_wee)
    t_w_end = rec_w_content.times[end]
    return t_end,t_w_end
end

##
mysim(0.5)

##

# run function one for precompilation
@time mysim(1.0)
@time mysim(10.0)

##

@profview mysim(10.0)

## allocation profiler!
# Collect an allocation profile
using Profile,PProf
Profile.Allocs.clear()
Profile.Allocs.@profile mysim(10.0)
# Export pprof allocation profile and open interactive profiling web interface.
PProf.Allocs.pprof()