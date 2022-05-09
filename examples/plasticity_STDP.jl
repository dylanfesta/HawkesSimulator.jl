
#=
# Spiking-based plasticity rules for two neurons

In this example, I show the effects of plasticity rules on a single
pre-post connection between the two neurons. The neural activity is
entirely regulated from the outside (e.g. the neurons do not interact
through their weights). This is to better illustrate plasticity rules
in their simplest form.

I also consider analytic results for purely Poisson firing (uncorrelated).

# Initialization
=#
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors,LaTeXStrings ; theme(:default) ; gr()

using InvertedIndices
using ProgressMeter
using Random
Random.seed!(0)

using HawkesSimulator; const global H = HawkesSimulator

## #src
#=
# Part 1: Poisson processes

Here the neurons are uncorrelated Poisson processes. 

The goal is to emphasize the rate-dependent part of plasticity, 
comparing pairwise STDP to triplet STDP 

## Plasticity: positive biased STPD rule 

Here the weight change will depent linearly on pre and post rate.

### Functions that generate spiketrains, plasticity, population, etc
=#
## #src


# this one generates the connection object
function do_2by2_connection(τplus::R,A2plus::R,τminus::R,A2minus::R;w_start::R=100.0) where R
  N=2
  _plast = HawkesSimulator.PairSTDP(τplus,τminus,A2plus,A2minus,N,N)
  W = fill(w_start,N,N)
  W[diagind(W)] .= 0.0
  return H.ConnectionWeights(W,_plast)
end

# this one generates spike trains, a network object, a weight recorder. It requres a connection
function population_trains(rate1::R,rate2::R,conn::H.Connection;
    ttot::R=5_000.0,krec::Integer=400) where R
  N = 2
  trains =[ H.make_poisson_samples(rate1,ttot),
             H.make_poisson_samples(rate2,ttot) ]
  _ps = H.PopulationState(H.InputUnit(H.SGTrains(trains)),N)  
  pop = H.PopulationInputTestWeights(_ps,conn)
  recweights = H.RecTheseWeights(conn.weights,ttot/krec,ttot)
  ntw = H.RecurrentNetworkExpKernel(pop,recweights)
  return recweights,ntw
end

# this put the previous two together
function plastic_two_neurons(τplus::R,A2plus::R,τminus::R,A2minus::R,rate1::R,rate2::R;
    w_start::R=100.0,ttot::R=5_000.,krec=400) where R
  conn = do_2by2_connection(τplus,A2plus,τminus,A2minus;w_start=w_start)
  recweights,ntw = population_trains( rate1,rate2,conn; ttot=ttot,krec=krec)
  return conn,recweights,ntw
end

# runs the network 
function run_network01(ntw,conn,recweights,ttot)
  H.reset!(ntw) 
  H.reset!(recweights)
  w_start = copy(conn.weights)
  _T=ttot-0.1
  t_now = 0.0
  prog = Progress(ceil(Integer,_T);dt=5) 
  while t_now <= _T
    t_now = H.dynamics_step_singlepopulation!(t_now,ntw)
    if t_now > _T
      ProgressMeter.finish!(prog)
    else
      update!(prog,floor(Integer,t_now))
    end
  end
  w_end = copy(conn.weights)
  return w_start,w_end,t_now
end

# Utility functions to make nice plots
function mynormalize(mat::Matrix{<:Real})
  _mi,_ma = extrema(mat)
  hasminus = _mi < -1E-4 
  if hasminus
    idxplus = mat .> 0
    matnorm = similar(mat)
    matnorm[idxplus] = mat[idxplus] ./ _ma
    matnorm[Not(idxplus)] = mat[Not(idxplus)] ./ abs(_mi)
  else
    matnorm = mat ./ _ma
  end
  return hasminus,matnorm
end

function plot_nice_DW(r1::AbstractVector{R},r2::AbstractVector{R},DW::Matrix{R}) where R
  rh = 0.5(r1[2]-r1[1])
  colorh = colorant"#F47D23"
  colorm = colorant"white"
  colorl = colorant"#147ABF"
  hasminus = minimum(DW) < -1E-2
  if hasminus
    _min,_max = extrema(DW)
    _mid = -_min/(_max-_min)
    myc = cgrad([colorl,colorm,colorh],[0,_mid,1.0])
  else
    myc = cgrad([colorm,colorh],[0,1.0])
  end
  _lims = (rh,rh+r1[end])
  return heatmap(r1,r2,DW;
    xlabel = L"r_{\mathrm{post}}",
    ylabel = L"r_{\mathrm{pre}}",
    xlims=_lims,ylims=_lims,
    ratio=1, color=myc)
end
function plot_nice_DW_normed(r1::AbstractVector{R},r2::AbstractVector{R},DW::Matrix{R}) where R
  colorh = colorant"#F47D23"
  colorm = colorant"white"
  colorl = colorant"#147ABF"
  hasminus,DWn = mynormalize(DW) 
  if hasminus
    myc = cgrad([colorl,colorm,colorh],[-1,0,1.0])
  else
    myc = cgrad([colorm,colorh],[0,1.0])
  end
  rh = 0.5(r1[2]-r1[1])
  _lims = (rh,rh+r1[end])
  return heatmap(r1,r2,DWn;
    xlabel = L"r_{\mathrm{post}}",
    ylabel = L"r_{\mathrm{pre}}",
    xlims=_lims,ylims=_lims,
    ratio=1,
    color=myc)
end


## #src
#=
###  Run numerical simulation for specific parameters 
=#
## src

const plast_eps = 1E-4
const A2plus = 1.0 * plast_eps
const τplus = 0.5
const A2minus = -0.7 * plast_eps
const τminus = 0.69
const rate1 = 10.
const rate2 = 15.0
const ttot = 1_000.0

# plot weight change in time

theplot = let  (conn,recweights,ntw) = 
    plastic_two_neurons(τplus,A2plus,τminus,A2minus,rate1,rate2;ttot=ttot),
  (w_start,w_end,t_end) = run_network01(ntw,conn,recweights,ttot)
  plt = plot(size=(300,200))
  ws12 = [_w[1,2] for _w in recweights.weights]
  ws21 = [_w[2,1] for _w in recweights.weights]
  wtimes = recweights.times
  plot!(plt,wtimes,ws12;label=L"$w_{12}$",linewidth=2)
  plot!(plt,wtimes,ws21;label=L"$w_{21}$",linewidth=2)
  plot!(plt,xlabel="time (s)",ylabel="weight",leg=:topleft,
    title="")
  plt
end
plot(theplot)
## #src
#=
The STDP rule is potentiation dominated, so the weights grow linearly.

### Run numerical simulation for varying rates
=#
## #src

const nrates = 20
const rates1 = range(0.1,45.;length=nrates)
const rates2 = copy(rates1)
const DW12 = Matrix{Float64}(undef,nrates,nrates)

const plast_eps = 1E-4
const A2plus = 1.0 * plast_eps
const τplus = 0.5
const A2minus = -0.7 * plast_eps
const τminus = 0.69
const ttot = 1_000.0

@showprogress for ij in CartesianIndices(DW12)
  conn,recweights,ntw = plastic_two_neurons(τplus,A2plus,τminus,A2minus,
    rates1[ij[1]],rates2[ij[2]];ttot=ttot)
  (w_start,w_end,t_end) = run_network01(ntw,conn,recweights,ttot)
  DW12[ij] = w_end[1,2] - w_start[1,2]
end

plot_nice_DW(rates1,rates2,DW12)


# compare with analytic values
rates_an = range(0.1,45.;length=150)
DW12_analytic_dense = let c = A2plus*τplus + A2minus*τminus
  (c*ttot) .* (rates_an * rates_an')
end

theplot = let  c = A2plus*τplus + A2minus*τminus
  DW12_analytic = (c*ttot) .* (rates1 * rates2')
  plt = plot()
  scatter!(plt,DW12[:],DW12_analytic[:],color=:black)
  plot!(plt,identity,ratio=1,xlabel="numeric",ylabel="analytic",linestyle=:dash,linewidth=3,
    color=:yellow,leg=false)
end

# show the analytic heatmap

plot_nice_DW_normed(rates_an,rates_an,DW12_analytic_dense)


## #src
#=
## Plasticity : triplet STDP with usual parameters
I will go straight to the part where I consider multiple rates
=#

## #src

# this one generates the connection object
function do_2by2_connection_triplets(τplus::R,A2plus::R,τminus::R,A2minus::R,
     τx::R,A3plus::R,τy::R,A3minus::R;w_start::R=100.0) where R
  N=2
  _plast = HawkesSimulator.PlasticityTriplets(τplus,τminus,τx,τy,A2plus,A3plus,A2minus,A3minus,N,N)
  W = fill(w_start,N,N)
  W[diagind(W)] .= 0.0
  return H.ConnectionWeights(W,_plast)
end


# this put the previous two together
function plastic_two_neurons_triplets(τplus::R,A2plus::R,τminus::R,A2minus::R,
    τx::R,A3plus::R,τy::R,A3minus::R,rate1::R,rate2::R;
    w_start::R=100.0,ttot::R=5_000.,krec=400) where R
  conn = do_2by2_connection_triplets(τplus,A2plus,τminus,A2minus,τx,A3plus,τy,A3minus;w_start=w_start)
  recweights,ntw = population_trains( rate1,rate2,conn; ttot=ttot,krec=krec)
  return conn,recweights,ntw
end


const nrates = 20
const rates1 = range(0.1,45.;length=nrates)
const rates2 = copy(rates1)
const DW12 = Matrix{Float64}(undef,nrates,nrates)

# parameters considered as "standard" for triplet STDP rule

const plast_eps = 1E-3

const	A2plus  = 7.5E-7 *plast_eps
const	A2minus = -7.0   *plast_eps
const	A3plus  = 6.0    *plast_eps  # 9.3
const	A3minus = -0.23  *plast_eps

const τplus = 17E-3
const τminus = 34E-3
const τy = 101E-3
const τx = 125E-3

# run simulation for varying rates
@showprogress for ij in CartesianIndices(DW12)
  conn,recweights,ntw = plastic_two_neurons_triplets(τplus,A2plus,τminus,A2minus,
    τx,A3plus,τy,A3minus,
    rates1[ij[1]],rates2[ij[2]];ttot=ttot)
  (w_start,w_end,t_end) = run_network01(ntw,conn,recweights,ttot)
  DW12[ij] = w_end[1,2] - w_start[1,2]
end

# show result
plot_nice_DW(rates1,rates2,DW12)

# compare with analytics

# compare with analytic values
rates_dense = range(0.1,45.;length=150)
DW12_analytic_dense = let r1r1 =  rates_dense * rates_dense'
  r2r1 = (rates_dense.^2)*rates_dense'
  r1r2 = rates_dense*(rates_dense.^2)'
  c11 = A2plus*τplus + A2minus*τminus
  c12 = A3minus*τminus*τx
  c21 = A3plus*τplus*τy  
  @. ttot * (c11*r1r1 + c12*r1r2 + c21*r2r1)
end

theplot = let r1r1 =  rates1 * rates2'
  r2r1 = (rates1.^2)*rates2'
  r1r2 = rates1*(rates2.^2)'
  c11 = A2plus*τplus + A2minus*τminus
  c12 = A3minus*τminus*τx
  c21 = A3plus*τplus*τy  
  DW12_analytic =  @. ttot * (c11*r1r1 + c12*r1r2 + c21*r2r1)
  plt = plot()
  scatter!(plt,DW12[:],DW12_analytic[:],color=:black)
  plot!(plt,identity,ratio=1,xlabel="numeric",ylabel="analytic",linestyle=:dash,linewidth=3,
    color=:yellow,leg=false)
end
plot(theplot)

# show the analytic heatmap

plot_nice_DW_normed(rates_an,rates_an,DW12_analytic_dense)


## #src
#=
# Part 2 : pairing protocol

Here  neurons are forced to spike as in typical STDP 
experimental protocols. That is, they both fire at the same
 rate, but with a time difference set to a certain $\Delta t$.
=#

## #src

function post_pre_spiketrains(rate::R,Δt_ro::R,Ttot::R;
    tstart::R = 0.05) where R
  post = collect(range(tstart,Ttot; step=inv(rate)))
  pre = post .- Δt_ro
  return [pre,post] 
end

# let's take a look at one example: rate of 5.0 Hz, 20 ms difference in spike time
_ = let trains = post_pre_spiketrains(5.0,20E-3,1.0)
  plt = plot()
  scatter!(plt,trains[1], 0.1 .+ zero(trains[1]),markersize=30,
      markercolor=:black,markershape=:vline,leg=false)
  scatter!(plt,trains[2], 0.2 .+ zero(trains[2]),markersize=30,
      markercolor=:black,markershape=:vline,leg=false)
  plot!(plt,ylims=(0.0,0.3),xlims=(0,1),xlabel="time (s)")
end

# Now I define a population of input neurons using this input protocol.
# The Population object includes the weight matrix and the 
# plasticity rules associated to it, both wrapped in the `Connection`
# object. I consider the connection as an input parameter of this function,
# to be set externally with the desired plasticity rule.
#
# Note that I define the population as `PopulationInputTestWeights` to
# indicate non-interacting weights
function post_pre_population(rate::Real,nreps::Integer,Δt_ro::Real,connection::H.Connection)
  Ttot = nreps/rate
  prepostspikes = post_pre_spiketrains(rate,Δt_ro,Ttot) 
  gen = H.SGTrains(prepostspikes)
  state = H.PopulationState(H.InputUnit(gen),2)
  return H.PopulationInputTestWeights(state,connection)
end

# ## Constructors
# Here I define a function that tests the plasticity rule.
# That is, given a $Δt$, it creates the input neurons, then
# it defines a network, and iterates it dynamically for 
# a certain number of pre-post repetitions.
# Finally it outputs the weight total change divided by time
# (so it's a weight change per second)
# 
# Since both neurons have the same plasticity rule, a 
# positive pre-post $\Delta t$ for neuron A, impacting the
#  $w_{\text{AB}}$ weight, is the 
# equivalent of a negative pre-post $\Delta t$ for neuron B, 
# impacting the $w_{\text{BA}}$ weight.

function test_stpd_rule(Δt::Real,connection::H.Connection;
    nreps=510,wstart=100.0,rate=0.5)
  num_spikes = nreps*2 - 10 # a bit less than total, for safety
  recorder = H.RecFullTrain(num_spikes)
  population = post_pre_population(rate,nreps,Δt,connection)
  network = H.RecurrentNetworkExpKernel(population,recorder)
  wmat = connection.weights
  fill!(wmat,wstart)
  wmat[diagind(wmat)] .= 0.0
  t_now = 0.0
  H.reset!.((network,recorder,connection)) # clear spike trains etc
  for _ in 1:num_spikes
    t_now = H.dynamics_step_singlepopulation!(t_now,network)
  end
  w12,w21 = wmat[1,2],wmat[2,1]
  H.reset!(network)
  dw12 = (w12-wstart)/t_now
  dw21 = (w21-wstart)/t_now
  return dw12,dw21
end

# ## STDP rule
# Here I define the plasticty type and the parameters that I want to test. 
# I choose the pairwise STDP rule.
# I initialize the weight matrix to 100.0 
# 
# Note once again that neurons are not interacting. The sole purpose
# of the "dummy" weights it to be changed by plasticity.

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

# let's do one run
dw12,dw21 = test_stpd_rule(2E-3,connection_test)
println("change in w12 : $(dw12), change in w21 $(dw21)")


# Neuron 1 spikes before neuron 2, therefore connection from 1 to 2, dw21 is 
# potentiated, while connection from 2 to 1, dw12 is depressed.

# ## STDP curve

# Compute and plot the weight changes due to STDP for varying $\Delta t$s

nsteps = 200
deltats = range(0.1E-3,100E-3;length=nsteps)
deltats_all = vcat(-reverse(deltats),deltats)
out = Vector{Float64}(undef,2*nsteps)

for (i,Δt) in enumerate(deltats)
  dw12,dw21 = test_stpd_rule(Δt,connection_test)
  out[nsteps+i]=dw21
  out[nsteps-i+1]=dw12
end


plot(deltats_all,out;leg=false,xlabel="Delta t",ylabel="dw/dt",linewidth=3)


# **THE END**

using Literate; Literate.markdown("examples/plasticity_STDP.jl","docs/src";documenter=true,repo_root_url="https://github.com/dylanfesta/HawkesSimulator.jl/blob/master") #src