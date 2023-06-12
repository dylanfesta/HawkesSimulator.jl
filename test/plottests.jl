using Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using QuadGK
using Plots,NamedColors ; theme(:dark) #; plotlyjs()
using FileIO
using BenchmarkTools
using Random
Random.seed!(0)

##

using  HawkesSimulator; const global H = HawkesSimulator

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

function absolute_errors(x::AbstractArray{<:Real},y::AbstractArray{<:Real})
  x,y=x[:],y[:]
  return abs.(x .-y )
end

function relative_errors(x::AbstractArray{<:Real},y::AbstractArray{<:Real})
  x,y=x[:],y[:]
  _m = @. max(abs(x),abs(y))
  return @. abs((x-y)/_m)
end

function rates_analytic(W::Matrix{R},r0::Vector{R}) where R
  return (I-W)\r0
end

## Test the rate-dependent component of symmetric and antisymmetric STDP rate-dependent rules




function run_simulation!(network,n_spikes;show_progress=false,t_start::Float64=0.0)
  t_now = t_start
  for _ in 1:n_spikes
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end

function ΔW_analytic_uncorrelated_symm(stpd_rule,rpost,rpre)
  αpre = stpd_rule.αpre
  αpost = stpd_rule.αpost
  θ = stpd_rule.θ
  bias = 2*(1+θ)
  return αpre*rpre + αpost*rpost + rpre*rpost*bias
end

function ΔW_analytic_uncorrelated_asymm(stpd_rule,rpost,rpre)
  αpre = stpd_rule.αpre
  αpost = stpd_rule.αpost
  θ = stpd_rule.θ
  bias = (1+θ)
  return αpre*rpre + αpost*rpost + rpre*rpost*bias
end



current_from_rates(rates::Vector{Float64},W::Matrix{Float64}) = (I-W)*rates
current_from_rates2D(re,ri,wie,wei) = 
  (ret= current_from_rates([re,ri],[ 0.0 -abs(wei) ; wie 0.0]) ; (ret[1],ret[2]))

function compute_weight_update_givenrate(re,ri,wie,wei,plasticity_rule)
  he,hi = current_from_rates2D(re,ri,wie,wei)
  return compute_weight_update_givencurrent(he,hi,wie,wei,plasticity_rule)
end

function compute_weight_update_givencurrent(he::R,hi::R,wie::R,wei::R,plasticity_rule;
    n_spikes=80_000,τker_e=50E-3,τker_i=30E-3) where R<:Real
  ne,ni = 1,1
  ps_e,tr_e = H.population_state_exp_and_trace(ne,τker_e)
  ps_i,tr_i = H.population_state_exp_and_trace_inhibitory(ni,τker_i)

  # Double connection: one not plastic, and one 
  # not interacting, to measure plasticity only
  conn_ie = H.ConnectionExpKernel(cat(wie;dims=2),tr_e)
  conn_ei = H.ConnectionExpKernel(cat(wei;dims=2),tr_i)
  conn_ei_for_plast = H.ConnectionNonInteracting(cat(100.0;dims=2),plasticity_rule)

# define populations
  pop_e = H.PopulationExpKernel(ps_e,[he,],(conn_ei,ps_i),(conn_ei_for_plast,ps_i))
  pop_i = H.PopulationExpKernel(ps_i,[hi,],(conn_ie,ps_e))
  # define recorder:  
  rec = H.RecFullTrain(n_spikes+1,2);
  # define network:
  netw = H.RecurrentNetworkExpKernel((pop_e,pop_i),(rec,));
  t_end = run_simulation!(netw,n_spikes)
  recc = H.get_content(rec)
  rate_e = H.numerical_rates(recc,ne,t_end;pop_idx=1)[1]
  rate_i = H.numerical_rates(recc,ni,t_end;pop_idx=2)[1]

  ΔW = (conn_ei_for_plast.weights[1,1] - 100.0) / (t_end*plasticity_rule.A) 
  return (ΔW=ΔW,rate_e=rate_e,rate_i=rate_i)
end


##
# Test 1 : all parameters!

ntest = 50

ΔW_test_an = Vector{Float64}(undef,ntest)
ΔW_test_num = similar(ΔW_test_an)

alp_pre = rand(Uniform(-3.,3.),ntest) 
alp_post = rand(Uniform(-3.,3.),ntest) 

theta_test = rand(Uniform(-4.,0.),ntest)
gamma_test = rand(Uniform(1.01,30.),ntest)
A_test = rand(Uniform(1E-8,1E-5),ntest)

rats_pre = rand(Uniform(1.0,30.),ntest)
rats_post = rand(Uniform(1.0,30.),ntest)

for k in 1:ntest
  _stdp_rule = H.PlasticitySymmetricSTDPX(A_test[k],theta_test[k],90E-3,gamma_test[k],
    alp_pre[k],alp_post[k],1,1)
  ΔW_test_an[k] = ΔW_analytic_uncorrelated_symm(_stdp_rule,rats_post[k],rats_pre[k])
  ΔW_test_num[k] = compute_weight_update_givenrate(rats_post[k],rats_pre[k],
    1E-6,1E-6,_stdp_rule).ΔW 
end

plotvs(ΔW_test_an,ΔW_test_num)
@test all(isapprox.(ΔW_test_an,ΔW_test_num,atol=10))

#############


ntest = 50

ΔW_test_an = Vector{Float64}(undef,ntest)
ΔW_test_num = similar(ΔW_test_an)

alp_pre = rand(Uniform(-3.,3.),ntest) 
alp_post = rand(Uniform(-3.,3.),ntest) 

theta_test = rand(Uniform(-4.,0.),ntest)
gamma_test = rand(Uniform(1.01,30.),ntest)
A_test = rand(Uniform(1E-8,1E-5),ntest)

rats_pre = rand(Uniform(1.0,30.),ntest)
rats_post = rand(Uniform(1.0,30.),ntest)

for k in 1:ntest
  _stdp_rule = H.PlasticityAsymmetricX(A_test[k],theta_test[k],90E-3,gamma_test[k],
    alp_pre[k],alp_post[k],1,1)
  ΔW_test_an[k] = ΔW_analytic_uncorrelated_asymm(_stdp_rule,rats_post[k],rats_pre[k])
  ΔW_test_num[k] = compute_weight_update_givenrate(rats_post[k],rats_pre[k],
    1E-6,1E-6,_stdp_rule).ΔW 
end

plotvs(ΔW_test_an,ΔW_test_num)

@test all(isapprox.(ΔW_test_an,ΔW_test_num,atol=10))