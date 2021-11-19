using ProgressMeter
using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using QuadGK
using Plots,NamedColors ; theme(:dark) ; plotlyjs()
using FileIO
using SparseArrays 
using BenchmarkTools
using FFTW
using Random
Random.seed!(0)

push!(LOAD_PATH, abspath(@__DIR__,".."))

using  HawkesSimulator; const global H = HawkesSimulator

##

N1 = 33
N2 = 12
weights1 = fill(Inf,N1,N1)
inputs1 = fill(Inf,N1)
weights2 = fill(Inf,N2,N2)
inputs2 = fill(Inf,N2)

tau =  123.5 
mykernel1 = H.KernelExp(tau) #considering single population
mykernel2 = H.KernelExp(rand()*tau) #considering single population

popstate1 = H.PopulationState(mykernel1,N1)
popstate2 = H.PopulationState(mykernel2,N2)

##
  network = H.RecurrentNetwork(popstate,weights,inputs)

  function simulate!(network,num_spikes)
    t_now = 0.0
    H.reset!(network) # clear spike trains etc
    for _ in 1:num_spikes
      t_now = H.dynamics_step!(t_now,network)
      H.flush_trains!(popstate,10.0;Tflush=5.0)
    end
    H.flush_trains!(popstate)
    return t_now
  end
  n_spikes = 200_000
  Tmax = simulate!(network,n_spikes)
  rates_ntw = H.numerical_rates(network.populations[1])
