```@meta
EditURL = "https://github.com/dylanfesta/HawkesSimulator.jl/blob/master/examples/plasticty_STDP.jl"
```

# Spiking-based plasticity rules under stimulation protocols (2 neurons)

In this example, I show the effects of plasticity rules on a single
pre-post connection between the two neurons. The neural activity is
entirely regulated from the outside (e.g. the neurons do not interact
through their weights). This is to better illustrate plasticity rules
in their simplest form.

## Initialization

````@example plasticty_STDP
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:default) ; gr()

using ProgressMeter
using Random
Random.seed!(0)

using HawkesSimulator; const global H = HawkesSimulator
````

Pairwise STPD

## Stimulation protocol

The neurons are forced to spike as in typical STDP measures. That is,
they both fire at the same rate, but with a time difference set
to a certain $\Delta t$.

````@example plasticty_STDP
function post_pre_spiketrains(rate::R,Δt_ro::R,Ttot::R;
    tstart::R = 0.05) where R
  post = collect(range(tstart,Ttot; step=inv(rate)))
  pre = post .- Δt_ro
  return [pre,post]
end
````

let's take a look at one example: rate of 5.0 Hz, 20 ms difference in spike time

````@example plasticty_STDP
_ = let trains = post_pre_spiketrains(5.0,20E-3,1.0)
  plt = plot()
  scatter!(plt,trains[1], 0.1 .+ zero(trains[1]),markersize=30,
      markercolor=:black,markershape=:vline,leg=false)
  scatter!(plt,trains[2], 0.2 .+ zero(trains[2]),markersize=30,
      markercolor=:black,markershape=:vline,leg=false)
  plot!(plt,ylims=(0.0,0.3),xlims=(0,1),xlabel="time (s)")
end
````

Now I define a population of input neurons using this input protocol.
The Population object includes the weight matrix and the
plasticity rules associated to it, both wrapped in the `Connection`
object. I consider the connection as an input parameter of this function,
to be set externally with the desired plasticity rule.

Note that I define the population as `PopulationInputTestWeights` to
indicate non-interacting weights

````@example plasticty_STDP
function post_pre_population(rate::Real,nreps::Integer,Δt_ro::Real,connection::H.Connection)
  Ttot = nreps/rate
  prepostspikes = post_pre_spiketrains(rate,Δt_ro,Ttot)
  gen = H.SGTrains(prepostspikes)
  state = H.PopulationState(H.InputUnit(gen),2)
  return H.PopulationInputTestWeights(state,connection)
end
````

## Constructors
Here I define a function that tests the plasticity rule.
That is, given a $Δt$, it creates the input neurons, then
it defines a network, and iterates it dynamically for
a certain number of pre-post repetitions.
Finally it outputs the weight total change divided by time
(so it's a weight change per second)

Since both neurons have the same plasticity rule, a
positive pre-post $\Delta t$ for neuron A, impacting the
 $w_{\text{AB}}$ weight, is the
equivalent of a negative pre-post $\Delta t$ for neuron B,
impacting the $w_{\text{BA}}$ weight.

````@example plasticty_STDP
function test_stpd_rule(Δt::Real,connection::H.Connection;
    nreps=510,wstart=100.0,rate=0.5)
  population = post_pre_population(rate,nreps,Δt,connection)
  network = H.RecurrentNetwork(population)
  wmat = connection.weights
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
````

## STDP rule
Here I define the plasticty type and the parameters that I want to test.
I choose the pairwise STDP rule.
I initialize the weight matrix to 100.0

Note once again that neurons are not interacting. The sole purpose
of the "dummy" weights it to be changed by plasticity.

````@example plasticty_STDP
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
````

let's do one run

````@example plasticty_STDP
dw12,dw21 = test_stpd_rule(2E-3,connection_test)
println("change in w12 : $(dw12), change in w21 $(dw21)")
````

Neuron 1 spikes before neuron 2, therefore connection from 1 to 2, dw21 is
potentiated, while connection from 2 to 1, dw12 is depressed.

## STDP curve

Compute and plot the weight changes due to STDP for varying $\Delta t$s

````@example plasticty_STDP
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
````

**THE END**

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

