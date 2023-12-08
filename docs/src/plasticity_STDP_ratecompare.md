```@meta
EditURL = "../../examples/plasticity_STDP_ratecompare.jl"
```

# Comparison of STDP rules in the rate-dominated regime

This example is both a proof of concept, and a way to test that ALL
my STDP rules are implemented correctly.

I will first consider uncorrelated neurons, and the ΔW over time.

Then I will consider an I to E connection, and check that converges
to the expected values.

This example also shows the different parametrizations that I
used for each rulle (all different! LOL)

# Initialization

````@example plasticity_STDP_ratecompare
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors,LaTeXStrings ; theme(:default) ; gr()

using InvertedIndices
using ProgressMeter
using Random
Random.seed!(0)

using HawkesSimulator; const global H = HawkesSimulator
````

# Utility functions

````@example plasticity_STDP_ratecompare
"""
  plotvs(x::AbstractArray{<:Real},y::AbstractArray{<:Real})

Plots X vs Y, and also the identity line.
"""
function plotvs(x::AbstractArray{<:Real},y::AbstractArray{<:Real})
  x,y=x[:],y[:]
  @info """
  The max differences between the two are $(extrema(x .-y ))
  """
  plt=plot()
  scatter!(plt,x,y;leg=false,ratio=1,color=:black,opacity=0.5)
  lm=xlims()
  plot!(plt,identity,range(lm...;length=3);linestyle=:dash,
    color=colorant"dark orange", opacity=0.5,linewidth=3)
  return plt
end

function absolute_errors(x::AbstractArray{<:Real},y::AbstractArray{<:Real})
  x,y=x[:],y[:]
  return abs.(x .-y )
end
function relative_errors(x::AbstractArray{<:Real},y::AbstractArray{<:Real})
  x,y=x[:],y[:]
  _m = @. max(abs(x),abs(y))
  return @. abs((x-y)/_m)
end
````

# Part 1 , uncorrelated pair of neurons

## General functions

````@example plasticity_STDP_ratecompare
function run_simulation!(network,n_spikes::Integer;t_start::Float64=0.0)
  t_now = t_start
  for _ in 1:n_spikes
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end

function compute_weight_update(hvec::Vector{R},plasticity_rule;
    wval = 1000.0,
    n_spikes::Integer=80_000,τker::Float64=50E-3) where R<:Real
  ps,tr = H.population_state_exp_and_trace(2,τker)
  wmat = [0.0 wval; wval 0.0]
  conn = H.ConnectionNonInteracting(wmat,plasticity_rule)
  pop = H.PopulationExpKernel(ps,hvec,(conn,ps))
  H.set_initial_rates!(pop,hvec)
  rec = H.RecFullTrain(n_spikes+1,1)
  netw = H.RecurrentNetworkExpKernel(pop,rec)
  t_end = run_simulation!(netw,n_spikes)
  recc = H.get_content(rec)
  rates = H.numerical_rates(recc,2,t_end;pop_idx=1)
  ΔW1 = (conn.weights[2,1] - wval) / t_end
  ΔW2 = (conn.weights[1,2] - wval) / t_end
  ΔW_avg = mean([ΔW1,ΔW2])
  return (ΔW1=ΔW1,ΔW2 = ΔW2,ΔW_avg = ΔW_avg,rates = rates)
end


#=  ## Start with Vogels et al. 2011

The average change of weight should be
$ B r_{\text{pre}}  r_{\text{post}} - \eta\, \alpha\,r_{\text{pre}}$.
Where the area under the curve  $B$ is $B = 2 \eta \tau$  and
$\alpha = 2 r_{\text{targ}}\, \tau$.

Putting them together we get:
```math
\Delta W = 2\;\eta\,\tau \, r_{\text{pre}} \; (r_{\text{post}} -  r_{\text{targ}})$
```
As expected, the weight is tracking the difference between the
target rate and the actual postsynaptic rate.
=#

function analyticΔW(plast::H.PlasticityInhibitory,
    rpre::Real,rpost::Real)
  rtarg = 0.5*plast.α/plast.τ
  return 2.0*plast.η*plast.τ*rpre*(rpost-rtarg)
end

function do_test_for_vogels(n::Integer;n_spikes::Integer=100_000)
  rpres = rand(Uniform(20.0,50.0),n)
  rposts = rand(Uniform(20.0,50.0),n)
  rtargs = rand(Uniform(10.0,80.0),n)
  τs = rand(Uniform(0.1,0.5),n)
  η = 1E-8
  anΔWs = Vector{Float64}(undef,n)
  numΔWs = similar(anΔWs)
  for k in 1:n
    plast = H.PlasticityInhibitory(τs[k],η,2,2;r_target=rtargs[k])
    anΔWs[k] = analyticΔW(plast,rpres[k],rposts[k])
    numΔWs[k] = compute_weight_update([rpres[k],rposts[k]],plast;
      wval=1E4,
      n_spikes=n_spikes).ΔW1
  end
  anΔWs ./= η
  numΔWs ./= η
  return (anΔWs=anΔWs,numΔWs=numΔWs)
end


out = do_test_for_vogels(80)
_ = let plt = plotvs(out.anΔWs,out.numΔWs)
  xlabel!(plt,"analytic ΔW")
  ylabel!(plt,"numerical ΔW")
  plt
end


#=  ## Now symmetric and antisymmetric STDPX rules

These are expanded versions of STDP that contain both rate-dependent
and correlation-dependent terms. Here we ignore the latter.

```math
\Delta W =  A \; \left(  \alpha_{\text{pre}} \, r_{\text{pre}} + \alpha_{\text{post}} \, r_{\text{post}} +
 B \, r_{\text{pre}} \, r_{\text{post}} \right)
```
With $ B= 2\;(1+\theta)$ for symmetric   and $ B = (1+\theta)$ for antisymmetric.
=#

function analyticΔW(plast::H.PlasticitySymmetricSTDPX,
    rpre::Real,rpost::Real)
  B = 2.0*(1.0+plast.θ)
  return plast.A * (plast.αpre*rpre + plast.αpost*rpost + B*rpre*rpost)
end

function analyticΔW(plast::H.PlasticityAsymmetricX,
    rpre::Real,rpost::Real)
  B = (1.0+plast.θ) # only difference with symmetric
  return plast.A * (plast.αpre*rpre + plast.αpost*rpost + B*rpre*rpost)
end

function do_test_for_symmetricX(n::Integer;n_spikes::Integer=100_000)
  rpres = rand(Uniform(20.0,50.0),n)
  rposts = rand(Uniform(20.0,50.0),n)
  A = 1E-6
  thetas = rand(Uniform(-1.0,0.0),n)
  τs = rand(Uniform(0.1,0.5),n)
  gammas = rand(Uniform(1.5,20.0),n)
  αpres = rand(Uniform(-3.0,3.0),n)
  αposts = rand(Uniform(-3.0,3.0),n)
  anΔWs = Vector{Float64}(undef,n)
  numΔWs = similar(anΔWs)
  for k in 1:n
    plast = H.PlasticitySymmetricSTDPX(A,thetas[k],τs[k],gammas[k],αpres[k],αposts[k],2,2)
    anΔWs[k] = analyticΔW(plast,rpres[k],rposts[k])
    numΔWs[k] = compute_weight_update([rpres[k],rposts[k]],plast;
      wval=1E4,
      n_spikes=n_spikes).ΔW1
  end
  anΔWs ./= A
  numΔWs ./= A
  return (anΔWs=anΔWs,numΔWs=numΔWs)
end


out = do_test_for_symmetricX(80)
_ = let plt = plotvs(out.anΔWs,out.numΔWs)
  xlabel!(plt,"analytic ΔW")
  ylabel!(plt,"numerical ΔW")
  plt
end
````

And the antisymmetric one. The function is the same, except for plasticity type
But γ now can also be less than 1.

````@example plasticity_STDP_ratecompare
function do_test_for_asymmetricX(n::Integer;n_spikes::Integer=100_000)
  rpres = rand(Uniform(20.0,50.0),n)
  rposts = rand(Uniform(20.0,50.0),n)
  A = 1E-6
  thetas = rand(Uniform(-1.0,0.0),n)
  τs = rand(Uniform(0.1,0.5),n)
  gammas = rand(Uniform(0.1,5.0),n)
  αpres = rand(Uniform(-3.0,3.0),n)
  αposts = rand(Uniform(-3.0,3.0),n)
  anΔWs = Vector{Float64}(undef,n)
  numΔWs = similar(anΔWs)
  for k in 1:n
    plast = H.PlasticityAsymmetricX(A,thetas[k],τs[k],gammas[k],αpres[k],αposts[k],2,2)
    anΔWs[k] = analyticΔW(plast,rpres[k],rposts[k])
    numΔWs[k] = compute_weight_update([rpres[k],rposts[k]],plast;
      wval=1E4,
      n_spikes=n_spikes).ΔW1
  end
  anΔWs ./= A
  numΔWs ./= A
  return (anΔWs=anΔWs,numΔWs=numΔWs)
end

out = do_test_for_asymmetricX(80)
_ = let plt = plotvs(out.anΔWs,out.numΔWs)
  xlabel!(plt,"analytic ΔW")
  ylabel!(plt,"numerical ΔW")
  plt
end



#=  ## Now symmetric and antisymmetric generalized STDP rules with leak terms

These rules are simular to the above, but contain and additional (constant) leak term.

Also, I slightly changed the parametrization, because.

```math
\Delta W =  Azero \; \left( \alpha_{0} +  \alpha_{\text{pre}} \, r_{\text{pre}} + \alpha_{\text{post}} \, r_{\text{post}} +
 B \, r_{\text{pre}} \, r_{\text{post}} \right)
```
With $B = (Aplus+Aminus)$ for both symmetric and antisymmetric.
So it's really the same equation for both.
=#


function analyticΔW(plast::Union{H.PlasticitySymmetricSTDPGL,H.PlasticityAsymmetricSTDPGL},
    rpre::Real,rpost::Real)
  B = plast.Aplus + plast.Aminus
  return plast.Azero * (plast.αzero + plast.αpre*rpre + plast.αpost*rpost + B*rpre*rpost)
end

function do_test_for_symmetricGL(n::Integer;n_spikes::Integer=100_000)
  rpres = rand(Uniform(20.0,50.0),n)
  rposts = rand(Uniform(20.0,50.0),n)
  Azero = 1E-6
  Aplusses = rand(Uniform(0.,0.3),n)
  Aminuses = rand(Uniform(-0.3,0.),n)
  τs = rand(Uniform(0.1,0.5),n)
  gammas = rand(Uniform(1.1,20.0),n)
  αzeros = rand(Uniform(-5.0,5.0),n)
  αpres = rand(Uniform(-3.0,3.0),n)
  αposts = rand(Uniform(-3.0,3.0),n)
  anΔWs = Vector{Float64}(undef,n)
  numΔWs = similar(anΔWs)
  for k in 1:n
    plast = H.PlasticitySymmetricSTDPGL(Azero,Aplusses[k],Aminuses[k],
      τs[k],gammas[k],αzeros[k],αpres[k],αposts[k],2,2)
    anΔWs[k] = analyticΔW(plast,rpres[k],rposts[k])
    numΔWs[k] = compute_weight_update([rpres[k],rposts[k]],plast;
      wval=1E4,
      n_spikes=n_spikes).ΔW1
  end
  anΔWs ./= Azero
  numΔWs ./= Azero
  return (anΔWs=anΔWs,numΔWs=numΔWs)
end

out = do_test_for_symmetricGL(80)
_ = let plt = plotvs(out.anΔWs,out.numΔWs)
  xlabel!(plt,"analytic ΔW")
  ylabel!(plt,"numerical ΔW")
  plt
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

