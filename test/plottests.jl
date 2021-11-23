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

# plasticity rule for two input neurons

# post-pre
function ps_trains(rate::R,Δt_ro::R,Ttot::R;
    tstart::R = 0.05) where R
  post = collect(range(tstart,Ttot; step=inv(rate)))
  pre = post .- Δt_ro
  return [pre,post] 
end

function get_test_pop(rate,nreps,Δt_ro,connection_test)
  Ttot = nreps/rate
  prepostspikes = ps_trains(rate,Δt_ro,Ttot) 
  gen = H.SGTrains(prepostspikes)
  state = H.PopulationState(H.InputUnit(gen),2)
  return H.PopulationInputTestWeights(state,connection_test)
end

connection_test = let wmat =  fill(100.0,2,2)
  wmat[diagind(wmat)] .= 0.0
  τplus = 10E-3
  τminus = 10E-3
  Aplus = 1E-1
  Aminus = -1E-1
  npost,npre = size(wmat)
  stdp_plasticity = H.PairSTDP(τplus,τminus,Aplus,Aminus,npost,npre)
  H.ConnectionWeights(wmat,stdp_plasticity)
end
nreps = 500
population = get_test_pop(0.5,nreps,2E-3,connection_test)

population.state.unittype.spike_generator.trains

network = H.RecurrentNetwork(population)

function simulate_plasticity_test!(network,num_spikes)
  wmat = connection_test.weights
  fill!(wmat,100)
  wmat[diagind(wmat)] .= 0.0
  t_now = 0.0
  H.reset!(network) # clear spike trains etc
  for _ in 1:num_spikes
    t_now = H.dynamics_step_singlepopulation!(t_now,network)
    H.plasticity_update!(t_now,network)
  end
  H.flush_trains!(network)
  return t_now
end
n_spikes = nreps*2-3
Tmax = simulate_plasticity_test!(network,n_spikes)


## test different deltas


function test_stpd_rule(Δt::Real,connection;nreps=510,wstart=100.0)
  population = get_test_pop(0.5,nreps,Δt,connection_test)
  network = H.RecurrentNetwork(population)
  wmat = connection_test.weights
  fill!(wmat,wstart)
  wmat[diagind(wmat)] .= 0.0
  t_now = 0.0
  H.reset!(network) # clear spike trains etc
  num_spikes = nreps*2 - 10 # a bit less than total, for safety
  for _ in 1:num_spikes
    t_now = H.dynamics_step_singlepopulation!(t_now,network)
    H.plasticity_update!(t_now,network)
  end
  w12,w21 = wmat[1,2],wmat[2,1]
  H.reset!(network)
  dw12 = (w12-wstart)/t_now
  dw21 = (w21-wstart)/t_now
  return dw12,dw21
end

connection_test = let wmat =  fill(100.0,2,2)
  wmat[diagind(wmat)] .= 0.0
  τplus = 10E-3
  τminus = 10E-3
  Aplus = 1E-1
  Aminus = 1E-1
  npost,npre = size(wmat)
  stdp_plasticity = H.PairSTDP(τplus,τminus,Aplus,Aminus,npost,npre)
  H.ConnectionWeights(wmat,stdp_plasticity)
end
_,_,dw12,dw21 = test_stpd_rule(1E-3,connection_test)

nsteps = 100
deltats = range(0.1E-3,100E-3;length=nsteps)
out = Vector{Float64}(undef,2*nsteps)

for (i,Δt) in enumerate(deltats)
  _,_,dw12,dw21 = test_stpd_rule(Δt,connection_test)
  out[nsteps+i]=dw21
  out[nsteps-i+1]=dw12
end

deltats_all = vcat(-reverse(deltats),deltats)

plot(deltats_all,out;leg=false,xlabel="Delta t",ylabel="dw/dt")




# # triplets version: post pre , post post
# function ps_trains(rate::R,Δt_ro::R,Δt_oo::R,Ttot::R;
#     tstart::R = 0.05,popτ::R=1E6) where R
#   _post1 = collect(range(tstart,Ttot-2Δt_oo; step=inv(rate)))
#   post = sort(vcat(_post1,_post1.+Δt_oo))
#   pre = _post1 .- Δt_ro
#   return S.PSFixedSpiketrain([pre,post],popτ)
# end
# ## generic connection pre to post (2<-1)
# wstart = sparse([ 0.0 0.0 ; 1000.0 0.0])
