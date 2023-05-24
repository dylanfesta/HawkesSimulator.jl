```@meta
EditURL = "https://github.com/dylanfesta/HawkesSimulator.jl/blob/master/examples/plasticity_rate_based.jl"
```

# Rate-based plasticity rules

Here I use the rate-dependent STPD rules for some simple examples.

# Initialization

````@example plasticity_rate_based
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors,LaTeXStrings ; theme(:default) ; gr()

using InvertedIndices
using ProgressMeter
using Random
Random.seed!(0)

using HawkesSimulator; const global H = HawkesSimulator
````

# Utility functions

````@example plasticity_rate_based
function do_colormap(minval::Real,maxval::Real;
     cminus=:red,czero=:white,cplus=:blue,
     ncolors::Integer=500)
  if minval >= 0
    return cgrad([czero,cplus], [0,1])
  end
  if maxval <= 0
    return cgrad([cminus,czero], [0,1])
  end
  _mid = minval/(minval-maxval)
  vcols = [cminus,fill(czero,ncolors-2)...,cplus]
  midrange = range(_mid-1E-10,_mid+1E-10,length=ncolors-2)
  vals = [0.,midrange...,1.]
  return cgrad(vcols,vals)
end

function Wplot(W::Matrix{<:Real},bounds::Tuple{<:Real,<:Real}=extrema(W);
    title::String="")
  N=size(W,1)
  _cmap = do_colormap(bounds...)
  return heatmap(W;ratio=1,
    xlims=(0,N).+0.5,
    ylims=(0,N).+0.5,
    xlabel = "pre",
    ylabel = "post",
    color=_cmap,
    clims=bounds,
    title=title)
end;
nothing #hide
````

function to run the siumulation

````@example plasticity_rate_based
function run_simulation!(network,n_spikes;show_progress=false,t_start::Float64=0.0)
  t_now = t_start
  pr = Progress(n_spikes;
    enabled=show_progress,dt=30.0,desc="Running network dynamics...")
  for _ in 1:n_spikes
    t_now = H.dynamics_step!(t_now,network)
    ProgressMeter.next!(pr)
  end
  ProgressMeter.finish!(pr)
  return t_now
end;
nothing #hide
````

# First example: E E network with symmetric rule in "tuning" plasticty regime

````@example plasticity_rate_based
n_spikes = 6_000_000
τker_e = 50E-3
ne = 10
he = 10.0
wmin = 1E-5

ps_e,tr_e = H.population_state_exp_and_trace(ne,τker_e);
nothing #hide
````

E to E plasticity is created here.
the tuning feature is regulated by setting θ clearly lower than -1

````@example plasticity_rate_based
plasticity_ee_nonblanket = let A = 8E-7,
  ree_targ = 11.0,
  τ = 40E-3,
  wee_max = 0.1, # will never be reached, as long as the target rate is low enough
  γ = 10.0,
  θ = -1.1, #  must be lower than -1 !
  αpre =  -ree_targ*(1+θ)
  αpost = -ree_targ*(1+θ)
  bounds=H.PlasticityBoundsLowHigh(wmin,wee_max)
  H.PlasticitySymmetricSTDPX(A,θ,τ,γ,αpre,αpost,ne,ne;
      plasticity_bounds=bounds)
end;
nothing #hide
````

generate initial weights

````@example plasticity_rate_based
W_start = fill(0.01,ne,ne)
W_start[diagind(W_start)] .= 0.0

connection_ee = H.ConnectionExpKernel(copy(W_start),tr_e,plasticity_ee_nonblanket);
nothing #hide
````

define populations

````@example plasticity_rate_based
pop_e = H.PopulationExpKernel(ps_e,fill(he,ne),(connection_ee,ps_e));
nothing #hide
````

Record all spikes for a single population

````@example plasticity_rate_based
rec_spk = H.RecFullTrain(n_spikes+1,1);
nothing #hide
````

Record weights too

````@example plasticity_rate_based
Δt_wrec = 5*60.0
Tend_wrec =  14*3600.0
rec_wee = H.RecTheseWeights(connection_ee.weights,Δt_wrec,Tend_wrec);
nothing #hide
````

define network:

````@example plasticity_rate_based
netw = H.RecurrentNetworkExpKernel((pop_e,),(rec_spk,rec_wee));
nothing #hide
````

Ready to run!

````@example plasticity_rate_based
H.reset!(netw)
t_end = run_simulation!(netw,n_spikes;show_progress=false)
@info """
Simulation completed at time $(round(t_end/60.0;digits=1)) min or $(round(t_end/3600.0;digits=2)) hours
"""


W_end = connection_ee.weights;
nothing #hide
````

get weights

````@example plasticity_rate_based
rec_weec = H.get_content(rec_wee)
times_W = rec_weec.times
Ws = rec_weec.weights;
nothing #hide
````

get spikes

````@example plasticity_rate_based
recc_spk = H.get_content(rec_spk)
trains_e = H.get_trains(recc_spk;pop_idx=1)

Δt_rates = 10*60.0
times_r, rates_insta_e = H.instantaneous_rates(trains_e,Δt_rates ,t_end);
nothing #hide
````

plot the rates to show that they are more or less stable

````@example plasticity_rate_based
plot(times_r ./ 60.0,rates_insta_e';
  label="rates",xlabel="time (min)",ylabel="rate (Hz)",linewidth=2,leg=false,
  linecolor=:blue,opacity=0.5,
  ylims=(0,maximum(rates_insta_e)*1.1))
````

plot the average weight, to check for stability

````@example plasticity_rate_based
theplot = let yplot = mean.(Ws)
  plot(times_W ./ 60.0,yplot;
    label="average weight",xlabel="time (min)",ylabel="weight",linewidth=5,leg=false,
    linecolor=colorant"orange",opacity=0.5,
    ylims=(0,maximum(yplot)*1.1 ))
end
````

plot the weight matrix before

````@example plasticity_rate_based
p1 = Wplot(W_start;title="initial weights")
````

plot weight after

````@example plasticity_rate_based
p2 = Wplot(W_end;title="final weights")
````

## Result

Even without heterosynaptic effects, then netwrok reaches a symmetric
sparse structre. The number of connections depends on the target rate.

But careful! When neuron correlate, they increase their target rate,
therefore the final rate will *not* correspond to the target rate,
but be higher! So a high target rate might result in an even higher
final rate, with the risk of triggering a runaway excitation.

In this regime, seems that the target rate should be very close to the
input current. So this type of plasticity can't be used to change
the network operating regine.

# Blanket exc to exc

We want something really rate-dominated. Therefore
the (1+θ) should be strong and negative.  Hence `θ = -4.0`

````@example plasticity_rate_based
he = 5.0
plasticity_ee_blanket = let A = 8E-7,
  ree_targ = 30.0,
  τ = 40E-3,
  wee_max = Inf, # here does not play a role!
  γ = 10.0,
  θ = -4.0, #  must be lower than -1 !
  αpre =  -ree_targ*(1+θ)
  αpost = -ree_targ*(1+θ)
  bounds=H.PlasticityBoundsLowHigh(wmin,wee_max)
  H.PlasticitySymmetricSTDPX(A,θ,τ,γ,αpre,αpost,ne,ne;
      plasticity_bounds=bounds)
end;
nothing #hide
````

generate initial weights

````@example plasticity_rate_based
W_start = fill(0.01,ne,ne)
W_start[diagind(W_start)] .= 0.0


H.reset!(ps_e)
connection_ee = H.ConnectionExpKernel(copy(W_start),tr_e,plasticity_ee_blanket);
nothing #hide
````

define populations

````@example plasticity_rate_based
pop_e = H.PopulationExpKernel(ps_e,fill(he,ne),(connection_ee,ps_e));
nothing #hide
````

Record all spikes for a single population

````@example plasticity_rate_based
rec_spk = H.RecFullTrain(n_spikes+1,1)
````

Record weights too

````@example plasticity_rate_based
rec_wee = H.RecTheseWeights(connection_ee.weights,Δt_wrec,Tend_wrec)
````

define network:

````@example plasticity_rate_based
netw = H.RecurrentNetworkExpKernel((pop_e,),(rec_spk,rec_wee));
nothing #hide
````

Ready to run!

````@example plasticity_rate_based
t_end = run_simulation!(netw,n_spikes;show_progress=false)
@info """
Simulation completed at time $(round(t_end/60.0;digits=1)) min or $(round(t_end/3600.0;digits=2)) hours
"""


W_end = connection_ee.weights;
nothing #hide
````

get weights

````@example plasticity_rate_based
rec_weec = H.get_content(rec_wee)
times_W = rec_weec.times
Ws = rec_weec.weights;
nothing #hide
````

get spikes

````@example plasticity_rate_based
recc_spk = H.get_content(rec_spk)
trains_e = H.get_trains(recc_spk;pop_idx=1)

times_r, rates_insta_e = H.instantaneous_rates(trains_e,Δt_rates ,t_end);
nothing #hide
````

plot the rates to show that they are more or less stable

````@example plasticity_rate_based
plot(times_r ./ 60.0,rates_insta_e';
  label="rates",xlabel="time (min)",ylabel="rate (Hz)",linewidth=2,leg=false,
  linecolor=:blue,opacity=0.5,
  ylims=(0,maximum(rates_insta_e)*1.1))
````

plot the average weight, to check for stability

````@example plasticity_rate_based
theplot = let yplot = mean.(Ws)
  plot(times_W ./ 60.0,yplot;
    label="average weight",xlabel="time (min)",ylabel="weight",linewidth=5,leg=false,
    linecolor=colorant"orange",opacity=0.5,
    ylims=(0,maximum(yplot)*1.1 ))
end
````

plot the weight matrix before

````@example plasticity_rate_based
p1 = Wplot(W_start;title="initial weights")
````

plot weight after

````@example plasticity_rate_based
p2 = Wplot(W_end;title="final weights")
````

## Result

The excitation tends to be much more distributed across the network,
and it is still symmetric.

# From symmetric to asymmetric STDP

An asymmetric STDP will force a directionality in the connection.
Let's see how that turns out. Again take the regime of
strong and sparse  so small (1+θ) and input current very close
to target rate.

````@example plasticity_rate_based
he = 5.0
plasticity_ee_asymm_blanket = let A = 8E-7,
  ree_targ = 5.01,
  τ = 40E-3,
  wee_max = 0.25,
  γ = 1.0,
  θ = -1.2,
  αpre =  -ree_targ*(1+θ)
  αpost = -ree_targ*(1+θ)
  bounds=H.PlasticityBoundsLowHigh(wmin,wee_max)
  H.PlasticityAsymmetricX(A,θ,τ,γ,αpre,αpost,ne,ne;
      plasticity_bounds=bounds)
end;

H.reset!(ps_e)
connection_ee = H.ConnectionExpKernel(copy(W_start),tr_e,plasticity_ee_asymm_blanket);
nothing #hide
````

define populations

````@example plasticity_rate_based
pop_e = H.PopulationExpKernel(ps_e,fill(he,ne),(connection_ee,ps_e));
nothing #hide
````

Record all spikes for a single population

````@example plasticity_rate_based
rec_spk = H.RecFullTrain(n_spikes+1,1);
nothing #hide
````

Record weights too

````@example plasticity_rate_based
rec_wee = H.RecTheseWeights(connection_ee.weights,Δt_wrec,Tend_wrec);
nothing #hide
````

define network:

````@example plasticity_rate_based
netw = H.RecurrentNetworkExpKernel((pop_e,),(rec_spk,rec_wee));
nothing #hide
````

Ready to run!

````@example plasticity_rate_based
t_end = run_simulation!(netw,n_spikes;show_progress=false);
@info """
Simulation completed at time $(round(t_end/60.0;digits=1)) min or $(round(t_end/3600.0;digits=2)) hours
"""


W_end = connection_ee.weights;
nothing #hide
````

get weights

````@example plasticity_rate_based
rec_weec = H.get_content(rec_wee)
times_W = rec_weec.times
Ws = rec_weec.weights;
nothing #hide
````

get spikes

````@example plasticity_rate_based
recc_spk = H.get_content(rec_spk)
trains_e = H.get_trains(recc_spk;pop_idx=1)

times_r, rates_insta_e = H.instantaneous_rates(trains_e,Δt_rates ,t_end);
nothing #hide
````

plot the rates to show that they are more or less stable

````@example plasticity_rate_based
plot(times_r ./ 60.0,rates_insta_e';
  label="rates",xlabel="time (min)",ylabel="rate (Hz)",linewidth=2,leg=false,
  linecolor=:blue,opacity=0.5,
  ylims=(0,maximum(rates_insta_e)*1.1))
````

plot the average weight, to check for stability

````@example plasticity_rate_based
theplot = let yplot = mean.(Ws)
  plot(times_W ./ 60.0,yplot;
    label="average weight",xlabel="time (min)",ylabel="weight",linewidth=5,leg=false,
    linecolor=colorant"orange",opacity=0.5,
    ylims=(0,maximum(yplot)*1.1 ))
end
````

plot the weight matrix before

````@example plasticity_rate_based
p1 = Wplot(W_start;title="initial weights")
````

plot weight after

````@example plasticity_rate_based
p2 = Wplot(W_end;title="final weights")
````

## Result

With an antisymmetric STDP, the system really likes saturation, but really dislikes symmetry.
The weight matrix is nearly perferctly antisymmetric, you could play Tetris with it.

**THE END**

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

