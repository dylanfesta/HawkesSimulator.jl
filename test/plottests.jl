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


function rates_analytic(W::Matrix{R},r0::Vector{R}) where R
  return (I-W)\r0
end

##
const ne = 1
const ni = 1
const N = ne+ni
const idxe = 1:ne
const idxi = idxe[end] .+ (1:ni)
const wmat = [ 10.  -15.
               10.  -15. ]
const wmat_ee = wmat[idxe,idxe]
const wmat_ie = wmat[idxi,idxe]
const wmat_ei = -wmat[idxe,idxi]
const wmat_ii = -wmat[idxi,idxi]


const τe = 1.
const τi = 0.6

const pse,trae = H.population_state_exp_and_trace(ne,τe)
const psi,trai = H.population_state_exp_and_trace_inhibitory(ni,τi)

const conn_ee = H.ConnectionExpKernel(wmat_ee,trae)
const conn_ie = H.ConnectionExpKernel(wmat_ie,trae)
const conn_ei = H.ConnectionExpKernel(wmat_ei,trai)
const conn_ii = H.ConnectionExpKernel(wmat_ii,trai)

const in_e = 120.0
const in_i = 120.0

const r0e = fill(in_e,ne)
const r0i = fill(in_i,ni)
const r0full = vcat(r0e,r0i)

const rates_an = rates_analytic(wmat,r0full)
@info """\\
  Expected E rates : $(round(rates_an[1];sigdigits=3))\\
  Expected I rates : $(round(rates_an[end];sigdigits=3))
"""
##

const population_e = H.PopulationExpKernel(pse,r0e,(conn_ei,psi),(conn_ee,pse))
const population_i = H.PopulationExpKernel(psi,r0i,(conn_ii,psi),(conn_ie,pse))

const n_spikes = 500_000
const recorder = H.RecFullTrain(n_spikes+1,2)
const network = H.RecurrentNetworkExpKernel((population_e,population_i),(recorder,))

function simulate!(network,num_spikes)
  t_now = 0.0
  H.reset!(network) # clear spike trains etc
  H.reset!.((pse,psi))
  @showprogress "Running Hawkes process..." for _ in 1:num_spikes
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end


t_end =simulate!(network,n_spikes)

trains = H.get_trains(recorder)

rates_num_e = H.numerical_rates(recorder)[1]
rates_num_i = H.numerical_rates(recorder)[2]

@info """\\
  Expected E rate : $(round(rates_an[1];sigdigits=3))\\
  Numerical E rate:  $(round(mean(rates_num_e[1:ne]);sigdigits=3))
  \n\n
  Expected I rate : $(round(rates_an[end];sigdigits=3))\\
  Numerical I rate:  $(round(mean(rates_num_i[1:ni]);sigdigits=3))
"""
