##
using BenchmarkTools
using Pkg
Pkg.activate(".")

using HawkesSimulator; global const H=HawkesSimulator
using LinearAlgebra

function rates_analytic(W::Matrix{R},r0::Vector{R}) where R
  return (I-W)\r0
end

function make_simple_wmat(ne::Integer,ni::Integer,
    wee::R,wie::R,wei::R,wii::R) where R<:Real
  ntot = ne+ni
  idxe = 1:ne
  idxi = ne .+ (1:ni)
  wmat = fill(0.0,(ntot,ntot))
  wmat[idxe,idxe].= abs(wee)/(ne-1)
  wmat[idxi,idxe].= abs(wie)/ne
  wmat[idxe,idxi].= -abs(wei)/ni
  wmat[idxi,idxi].= -abs(wii)/(ni-1)
  wmat[diagind(wmat)] .= 0.0
  return wmat
end

## Just do a 80 E 20 I  network
# exp kernel, no plasticity, nothing fancy

ne = 80
ni = 20
τe,τi = 200E-3,100E-3

ntot = ne+ni

wee = 1.4
wie = 1.0
wei = -1.0
wii = -0.1

h_e,h_i = 60.,50.0

wmat = make_simple_wmat(ne,ni,wee,wie,wei,wii)

idxe = 1:ne
idxi = ne .+ (1:ni)
wmat_ee = wmat[idxe,idxe]
wmat_ie = wmat[idxi,idxe]
wmat_ei = abs.(wmat[idxe,idxi])
wmat_ii = abs.(wmat[idxi,idxi])


pse,trae = H.population_state_exp_and_trace(ne,τe)
psi,trai = H.population_state_exp_and_trace_inhibitory(ni,τi)

conn_ee = H.ConnectionExpKernel(wmat_ee,trae)
conn_ie = H.ConnectionExpKernel(wmat_ie,trae)
conn_ei = H.ConnectionExpKernel(wmat_ei,trai)
conn_ii = H.ConnectionExpKernel(wmat_ii,trai)


r0e = fill(h_e,ne)
r0i = fill(h_i,ni)
r0full = vcat(r0e,r0i)

rates_an = rates_analytic(wmat,r0full)

population_e = H.PopulationExpKernel(pse,r0e,(conn_ei,psi),(conn_ee,pse);
  nonlinearity=H.NLRmax(300.))
population_i = H.PopulationExpKernel(psi,r0i,(conn_ii,psi),(conn_ie,pse);
  nonlinearity=H.NLRmax(300.))

n_spikes = 5_000
recorder = H.RecFullTrain(n_spikes+1,2)
network = H.RecurrentNetworkExpKernel((population_e,population_i),(recorder,))


function simulate!(network,num_spikes;initial_e=nothing,initial_i=nothing)
  t_now = 0.0
  H.reset!(network) # clear spike trains etc
  H.reset!.((pse,psi))
  H.set_initial_rates!(population_e,initial_e)
  H.set_initial_rates!(population_i,initial_i)
  for _ in 1:num_spikes
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end
##

##

_ = simulate!(network,10;initial_e=r0e,initial_i=r0i)


@benchmark t_end = simulate!($network,$n_spikes;initial_e=$r0e,initial_i=$r0i)
# 20230315 (plastic)
#BenchmarkTools.Trial: 8 samples with 1 evaluation.
# Range (min … max):  601.469 ms … 741.320 ms  ┊ GC (min … max): 10.78% … 10.69%
# Time  (median):     646.137 ms               ┊ GC (median):    10.80%
# Time  (mean ± σ):   653.585 ms ±  44.753 ms  ┊ GC (mean ± σ):  10.63% ±  0.70%

# Memory estimate: 505.65 MiB, allocs estimate: 10786010.