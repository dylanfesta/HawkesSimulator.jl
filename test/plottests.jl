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

nneus = 2
tauker = 0.5
trace_ker = H.Trace(tauker,nneus)
trace_boh = H.Trace(123.0,nneus)

popstate = H.PopulationStateExpKernel(nneus,trace_ker,trace_boh)
myweights = [0.31 -0.3
            0.9  -0.15]
myinputs = [5.0 , 5.0]
rates_analytic  = inv(I-myweights)*myinputs

connection = H.ConnectionExpKernel(myweights,trace_ker)
population = H.PopulationExpKernel(popstate,connection,myinputs)

n_spikes = 10_000

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


##
error("shtop!")

##

##

function generate_poisson_train(rate::R,Tend::R;Tstart=0.0) where R
  # preallocate for efficiency (even if it's already super fast)
  nmax = round(Integer,rate*Tend*1.5) 
  ret = Vector{Float64}(undef,nmax)
  k = 0
  t_spike_now = 0.0
  while t_spike_now <= Tend
    k+=1
    t_spike_now += -log(rand())/rate # add delta t
    ret[k] = t_spike_now
  end
  deleteat!(ret,k+1:nmax) # remove unnecessary elements
  if !iszero(Tstart)
    ret .+= Tstart # shift by start time, if needed
  end
  return ret
end

function generate_perturbed_train1(rate::R,dissimilarity::R,
    reference_train::Vector{R}) where R
  @assert 0.0 <= dissimilarity <= 1.0
  new_train = copy(reference_train)
  nspikes = length(reference_train)
  nchange = round(Integer,dissimilarity*nspikes) # how many spikes to remove?
  if iszero(nchange)
    return new_train
  end
  # select spikes to delete, and delete them
  idx_change = sample(1:nspikes,nchange;replace=false,ordered=true)
  deleteat!(new_train,idx_change)
  rate_add = rate*dissimilarity
  Tend = nspikes/rate
  # add replacement spikes
  new_spikes = generate_poisson_train(rate_add,Tend)
  new_train = sort!(vcat(new_train,new_spikes))
  return new_train
end

function quickrate(train::Vector{R},Tend::R) where R
  k = findfirst(>=(Tend),train)
  if isnothing(k)
    error("please set Tend correctly!")
  end
  return (k-1)/Tend
end

##