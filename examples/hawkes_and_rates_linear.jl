#=
Compare a 2D E/I linear model to the corresponding Hawkes process
=#

using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) ; plotlyjs()
using FFTW

using ProgressMeter
using Random
Random.seed!(0)

using HawkesSimulator; const global H = HawkesSimulator

function onedmat(x::Real)
  return fill(x,(1,1))
end 


## rate model first 


function iofunction(v::R,_v0::R,_α::R) where R<:Real
  return max(_α*(v-_v0),zero(R))
end
function iofunction_inv(r::R,_v0::R,_α::R) where R<:Real
  return max(zero(R),r/_α+_v0)
end

global const v0 = -70.0 # -70
global const α = 0.03 # 0.3
global const taus = [20.0, 10.0].*1E-3
global const weights = [1.25 -0.65 ; 1.2 -0.5] 
global const input = [50.0, 50.0] 
global const dt = 0.01E-3
global const Ttot = 1.0
global const taus_hawk = [3.0,3.0]

##

_ = let vs = range(-90,50.,length=150)
  rs = @. iofunction(vs,v0,α)
  plot(vs,rs;linewidth=2,leg=false)
end

##

function euler_dv(v::Vector{R},_weights::Matrix{R},_input::Vector{R},_taus::Vector{R},_v0::R,_α::R) where R
  r = @. iofunction(v,_v0,_α)
  return (.-v .+ (_weights*r) .+ _input) ./ _taus
end

function run_2D_network()
  v_start_e,v_start_i = 0.0,0.0   
  times = range(0.0,Ttot;step=dt)   
  ntimes = length(times)
  rates = Matrix{Float64}(undef,2,ntimes)
  v = [v_start_e,v_start_i]
  for tt in eachindex(times)
    rates[:,tt] = iofunction.(v,v0,α)
    dv = euler_dv(v,weights,input,taus,v0,α)
    v += dv*dt
  end
  return times,rates
end

##

runtimes,runrates = run_2D_network()


## Now the Hawkes process
weights_equiv = α^2 .* weights
inputs_equiv =  α .* (input .- (α .* weights+I)*fill(v0,2))

pop_e,pop_i = H.PopulationExp(taus_hawk[1]),H.PopulationExp(taus_hawk[2])

ps_e,ps_i = let _inp = inputs_equiv
  H.PopulationState(pop_e,[_inp[1]]),
  H.PopulationState(pop_i,[_inp[2]])
end

inntw_e,inntw_i = let _weights = weights_equiv
   H.InputNetwork(ps_e,[ps_e, ps_i],[onedmat(_weights[1,1]),onedmat(_weights[1,2])]),
   H.InputNetwork(ps_i,[ps_e, ps_i],[onedmat(_weights[2,1]),onedmat(_weights[2,2])])
end

function run_simulation!(netws,nspikes)
  t_now = 0.0
  H.reset!.(netws) # clear spike trains etc
  @showprogress 1.0 "Running Hawkes process..." for k in 1:nspikes
    t_now = H.dynamics_step!(t_now,netws)
    if k%1_000 == 0
      for netw in netws
        H.clear_trains!(netw.postpops) 
      end
    end
  end
  for netw in netws
    H.clear_trains!(netw.postpops,-1.0) 
  end
  return t_now
end
##
nspikes = 50_000
t_end=run_simulation!([inntw_e,inntw_i],nspikes)

##

rates_hawkes = [ H.numerical_rates(ps_e)[1],H.numerical_rates(ps_i)[1]]

@info "final rates are $(runrates[:,end])"
@info "Hawkes final rates are $rates_hawkes"
##