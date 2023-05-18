#=
# Rate-based plasticity rules 

Here I use the rate-dependent STPD rules for some simple examples.

# Initialization
=#


using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors,LaTeXStrings ; theme(:default) ; gr()

using InvertedIndices
using ProgressMeter
using Random
Random.seed!(0)

using HawkesSimulator; const global H = HawkesSimulator

# # Utility functions


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

# function to run the siumulation
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

## #src
# # First example: E E network with symmetric rule in "tuning" plasticty regime

n_spikes = 3_000_000
τker_e = 50E-3
ne = 10
he = 20.0
wmin = 1E-5
  
ps_e,tr_e = H.population_state_exp_and_trace(ne,τker_e);

# E to E plasticity is created here.
# the tuning feature is regulated by setting θ clearly lower than -1
plasticity_ee_nonblanket = let A = 2E-6,
  ree_targ = 30.0,
  τ = 40E-3,
  wee_max = 1.1, # will never be reached, as long as the target rate is low enough
  γ = 10.0,
  θ = -3.0, #  must be lower than -1 !
  αpre =  0.0
  αpost = -ree_targ*(1+θ)
  bounds=H.PlasticityBoundsLowHigh(wmin,wee_max)
  H.PlasticitySymmetricSTDPX(A,θ,τ,γ,αpre,αpost,ne,ne;
      plasticity_bounds=bounds)
end;

# generate initial weights
W_start = rand(ne,ne)*0.01 .+ wmin
W_start[diagind(W_start)] .= 0.0

connection_ee = H.ConnectionExpKernel(copy(W_start),tr_e,plasticity_ee_nonblanket);

# define populations
pop_e = H.PopulationExpKernel(ps_e,fill(he,ne),(connection_ee,ps_e));

# Record all spikes for a single population
rec_spk = H.RecFullTrain(n_spikes+1,1);

# Record weights too
Δt_wrec = 30.0
Tend_wrec =  4*3600.0
rec_wee = H.RecTheseWeights(connection_ee.weights,Δt_wrec,Tend_wrec);

# define network:
netw = H.RecurrentNetworkExpKernel((pop_e,),(rec_spk,rec_wee));

## #src

# Ready to run!
t_end = run_simulation!(netw,n_spikes;show_progress=false)
@info """
Simulation completed at time $(round(t_end/60.0;digits=1)) min or $(round(t_end/3600.0;digits=2)) hours
"""

## #src

W_end = connection_ee.weights;

# get weights
rec_weec = H.get_content(rec_wee)
times_W = rec_weec.times
Ws = rec_weec.weights;

# get spikes
recc_spk = H.get_content(rec_spk)
trains_e = H.get_trains(recc_spk;pop_idx=1)

Δt_rates = 10.0
times_r, rates_insta_e = H.instantaneous_rates(trains_e,Δt_rates ,t_end);

# plot the rates to show that they are more or less stable
plot(times_r ./ 60.0,rates_insta_e';
  label="rates",xlabel="time (min)",ylabel="rate (Hz)",linewidth=2,leg=false,
  linecolor=:blue,opacity=0.5)

# plot the average weight, to check for stability
plot(times_W ./ 60.0,mean.(Ws);
  label="average weight",xlabel="time (min)",ylabel="weight",linewidth=2,leg=false,
  linecolor=:blue,opacity=0.5)

p1 = Wplot(W_start;title="initial weights")


p2 = Wplot(W_end;title="final weights")

## #src

#=
## Result

Note that a mutual saturating connection would be enough to trigger exploding runaway
 excitation. However, the system is self-stabilizing, final weights are well below saturation.

Due to the parameter choice, few neurons have the most
outgoing connections. With even stronger tuning (like θ = -5.0) a single neuron
will be  driving the rest of the network.


# Blanket exc to exc

I set θ just a tiny bit below -1

=#
## #src
plasticity_ee_blanket = let A = 2E-6,
  ree_targ = 30.0,
  τ = 40E-3,
  wee_max = 1.1,
  γ = 10.0,
  θ = -1.2, #  must be lower than -1 !
  αpre =  0.0
  αpost = -ree_targ*(1+θ)
  bounds=H.PlasticityBoundsLowHigh(wmin,wee_max)
  H.PlasticitySymmetricSTDPX(A,θ,τ,γ,αpre,αpost,ne,ne;
      plasticity_bounds=bounds)
end;


# generate initial weights
W_start = rand(ne,ne)*0.01 .+ wmin
W_start[diagind(W_start)] .= 0.0

H.reset!(ps_e)
connection_ee = H.ConnectionExpKernel(copy(W_start),tr_e,plasticity_ee_blanket);
# define populations
pop_e = H.PopulationExpKernel(ps_e,fill(he,ne),(connection_ee,ps_e));

# Record all spikes for a single population
rec_spk = H.RecFullTrain(n_spikes+1,1)

# Record weights too
Δt_wrec = 30.0
Tend_wrec =  3*3600.0
rec_wee = H.RecTheseWeights(connection_ee.weights,Δt_wrec,Tend_wrec)

# define network:
netw = H.RecurrentNetworkExpKernel((pop_e,),(rec_spk,rec_wee));

## #src

# Ready to run!
t_end = run_simulation!(netw,n_spikes;show_progress=false)
@info """
Simulation completed at time $(round(t_end/60.0;digits=1)) min or $(round(t_end/3600.0;digits=2)) hours
"""

## #src

W_end = connection_ee.weights;

# get weights
rec_weec = H.get_content(rec_wee)
times_W = rec_weec.times
Ws = rec_weec.weights;

# get spikes
recc_spk = H.get_content(rec_spk)
trains_e = H.get_trains(recc_spk;pop_idx=1)

Δt_rates = 10.0
times_r, rates_insta_e = H.instantaneous_rates(trains_e,Δt_rates ,t_end);

# plot the rates to show that they are more or less stable
plot(times_r ./ 60.0,rates_insta_e';
  label="rates",xlabel="time (min)",ylabel="rate (Hz)",linewidth=2,leg=false,
  linecolor=:blue,opacity=0.5)

# plot the average weight, to check for stability
plot(times_W ./ 60.0,mean.(Ws);
  label="average weight",xlabel="time (min)",ylabel="weight",linewidth=2,leg=false,
  linecolor=:blue,opacity=0.5)

# plot the weight matrix before and after

p1 = Wplot(W_start;title="initial weights (very low)")

p2 = Wplot(W_end;title="final weights")

## #src

#=
## Result

The excitation tends to be much more distributed across the network, and is symmetric,
matching the symmetry of the pairwise STDP component.
=#


## #src

#=

# From symmetric to asymmetric STDP

An asymmetric STDP will force a certain directionality in the connection.
Let's see how that turns out.

=#
## #src
plasticity_ee_asymm_blanket = let A = 2E-6,
  ree_targ=20.,
  τ = 40E-3,
  wee_max = 0.3,
  γ = 1.0,
  θ = -1.3, # try either -1.3 or  -1.02 
  αpre =  0.0
  αpost = -ree_targ*(1+θ)
  bounds=H.PlasticityBoundsLowHigh(wmin,wee_max)
  H.PlasticityAsymmetricX(A,θ,τ,γ,αpre,αpost,ne,ne;
      plasticity_bounds=bounds)
end;

H.reset!(ps_e)
connection_ee = H.ConnectionExpKernel(copy(W_start),tr_e,plasticity_ee_asymm_blanket);
# define populations
pop_e = H.PopulationExpKernel(ps_e,fill(he,ne),(connection_ee,ps_e));

# Record all spikes for a single population
rec_spk = H.RecFullTrain(n_spikes+1,1);

# Record weights too
Δt_wrec = 30.0
Tend_wrec =  3*3600.0
rec_wee = H.RecTheseWeights(connection_ee.weights,Δt_wrec,Tend_wrec);

# define network:
netw = H.RecurrentNetworkExpKernel((pop_e,),(rec_spk,rec_wee));

## #src

# Ready to run!
t_end = run_simulation!(netw,n_spikes;show_progress=false);
@info """
Simulation completed at time $(round(t_end/60.0;digits=1)) min or $(round(t_end/3600.0;digits=2)) hours
"""

## #src

W_end = connection_ee.weights;

# get weights
rec_weec = H.get_content(rec_wee)
times_W = rec_weec.times
Ws = rec_weec.weights;

# get spikes
recc_spk = H.get_content(rec_spk)
trains_e = H.get_trains(recc_spk;pop_idx=1)

Δt_rates = 10.0
times_r, rates_insta_e = H.instantaneous_rates(trains_e,Δt_rates ,t_end);

# plot the rates to show that they are more or less stable
plot(times_r ./ 60.0,rates_insta_e';
  label="rates",xlabel="time (min)",ylabel="rate (Hz)",linewidth=2,leg=false,
  linecolor=:blue,opacity=0.5)

# plot the average weight, to check for stability
plot(times_W ./ 60.0,mean.(Ws);
  label="average weight",xlabel="time (min)",ylabel="weight",linewidth=2,leg=false,
  linecolor=:blue,opacity=0.5)

# plot the weight matrix before and after

p1 = Wplot(W_start;title="initial weights (very low)")

p2 = Wplot(W_end;title="final weights")

## #src

#=
## Result

With an antisymmetric STDP, the system really likes saturation, but really dislikes symemtry.
I had to further lower the target rate, and to set a lower bound on the weights, that now is almost 
always reached.  With  `θ = -1.02` we have a "blanket" regime, where most neurons are interccnnected.
With `θ = -1.3` few "winning" neurons give input to the rest of the network.
=#


# **THE END**

## publish in documentation #src
thisfile = joinpath(splitpath(@__FILE__)[end-1:end]...) #src
using Literate; Literate.markdown(thisfile,"docs/src";documenter=true,repo_root_url="https://github.com/dylanfesta/HawkesSimulator.jl/blob/master") #src