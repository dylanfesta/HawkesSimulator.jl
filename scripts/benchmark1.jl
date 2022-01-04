# compare exponential kernel specialized system with 
# exponential kernel in unspecialized system

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

nneus = 2
tauker = 0.5
trace_ker = H.Trace(tauker,nneus,H.ForDynamics())
trace_useless = H.Trace(123.0,nneus,H.ForPlasticity())

popstate = H.PopulationStateExpKernel(nneus,trace_ker,trace_useless)
myweights = [0.31 -0.3
            0.9  -0.15]
myinputs = [5.0 , 5.0]
rates_analytic  = inv(I-myweights)*myinputs

connection = H.ConnectionExpKernel(myweights,trace_ker)
population = H.PopulationExpKernel(popstate,connection,myinputs)

n_spikes = 10_000
recorder = H.RecFullTrain(n_spikes,1)
network = H.RecurrentNetworkExpKernel(population,recorder)

function simulate1!(network,num_spikes)
  t_now = 0.0
  H.reset!(network) # clear spike trains etc
  for _ in 1:num_spikes
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end
function simulate2!(network,num_spikes)
  t_now = 0.0
  H.reset!(network) # clear spike trains etc
  for _ in 1:num_spikes
    t_now = H.dynamics_step_singlepopulation_multi!(t_now,network)
  end
  return t_now
end
##
@benchmark simulate1!(network,n_spikes)
##
@benchmark simulate2!(network,n_spikes)
