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
function plotvs(x::AbstractArray{<:Real},y::AbstractArray{<:Real})
  x,y=x[:],y[:]
  @info """
  The max differences between the two are $(extrema(x .-y ))
  """
  plt=plot()
  scatter!(plt,x,y;leg=false,ratio=1,color=:white)
  lm=xlims()
  plot!(plt,identity,range(lm...;length=3);linestyle=:dash,color=:yellow)
  return plt
end

##

nneus = 3
tauker = 1.5
rates_start = 50.0
Tend = 60.0

noweights = zeros(nneus,nneus)
noinputs = zeros(nneus)

## generate trains
trains = [ H.make_poisson_samples(rates_start,Tend) for _ in 1:nneus]

## Build the population 

trace_ker = H.Trace(tauker,nneus,H.ForDynamics())


popstate = H.PopulationStateMixedExp(nneus,trains,trace_ker)
connection = H.ConnectionExpKernel(noweights,trace_ker)
population = H.PopulationMixedExp(popstate,connection,noinputs)

n_spikes = round(Integer,rates_start*Tend*0.9*nneus)
recorder = H.RecFullTrain(n_spikes,1)
network = H.RecurrentNetworkExpKernel(population,recorder)

function simulate!(network,num_spikes)
  t_now = 0.0
  H.reset!(network) # clear spike trains etc
  for _ in 1:num_spikes
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end

Tmax = simulate!(network,n_spikes)

trains_out = H.get_trains(recorder,nneus)

@test all( trains[1][1:length(trains_out[1])] .== trains_out[1])
@test all( trains[2][1:length(trains_out[2])] .== trains_out[2])
@test all( trains[3][1:length(trains_out[3])] .== trains_out[3])

##


someweights = [  0 0 0 ; 0.5 0 0 ; 1.0 0 0.0]

popstate = H.PopulationStateMixedExp(nneus,trains,trace_ker)
connection = H.ConnectionExpKernel(someweights,trace_ker)
population = H.PopulationMixedExp(popstate,connection,noinputs)

n_spikes = round(Integer,(1+1.5+2)rates_start*Tend*0.9)
recorder = H.RecFullTrain(n_spikes,1)
network = H.RecurrentNetworkExpKernel(population,recorder)

Tmax = simulate!(network,n_spikes)

trains_out2 = H.get_trains(recorder,nneus)

therates = H.numerical_rates(recorder,nneus,Tmax)

@test all( trains[1][1:length(trains_out2[1])] .== trains_out2[1])
@test isapprox(therates[2],1.5*rates_start;rtol=0.2)
@test isapprox(therates[3],2*rates_start;rtol=0.2)


##
#error("shtop!")