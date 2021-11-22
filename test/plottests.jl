using ProgressMeter
using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using QuadGK
using Plots,NamedColors ; theme(:dark) #; plotlyjs()
using FileIO
using SparseArrays 
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
population1 = H.Population(popstate1,inputs1,
  (H.ConnectionWeights(weights1),popstate1) )

population1 = H.Population(popstate1,inputs1,
  (H.ConnectionWeights(weights1),popstate1) )

population2 = H.Population(popstate2,inputs2,
  (H.ConnectionWeights(weights2),popstate2) )

# multi-pop constructor
network = H.RecurrentNetwork(population1,population2)

##

target_rates1 = rand(Uniform(5.,12.),N1)
target_rates2 = rand(Uniform(12.,44.),N2)
target_rates = [target_rates1,target_rates2]

Ttot = 500.0

H.do_warmup!(Ttot,network,target_rates)
H.flush_trains!(network)

@test all( isapprox.(H.numerical_rates(population1),target_rates1;rtol=0.2) )
@test all( isapprox.(H.numerical_rates(population2),target_rates2;rtol=0.2) )


##

H.numerical_rates(network.populations[1])

plotvs(H.numerical_rates(population1),target_rates1  )
plotvs(H.numerical_rates(population2),target_rates2  )

##