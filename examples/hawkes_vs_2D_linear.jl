# # Compares a 2D Hawekes process to a 2D linear system

#=
In this example, I show that the mean rates in a 2D Hawes process
perfectly match the mean rates of a 2D rate model. 

I use exponential synaptic kernels (but it is valid for any kernel, since
the kernel shape does not influence the mean rate)
=#

# ## Initialization

using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:default)
using FFTW

using ProgressMeter
using Random
Random.seed!(0)

using HawkesSimulator; const global H = HawkesSimulator

function onedmat(x::Real)
  return cat(x;dims=2)
end; 

## #src
# ## Rate Model

function iofunction(v::R,_v0::R,_α::R) where R<:Real
  return max(_α*(v-_v0),zero(R))
end
function iofunction_inv(r::R,_v0::R,_α::R) where R<:Real
  return max(zero(R),r/_α+_v0)
end

const v0 = -70.0 # -70
const α = 0.03 # 0.3
const taus = [20.0, 10.0].*1E-3
const weights = [1.25 -0.65 ; 1.2 -0.5] 
const input = [20.0, 50.0] 
const dt = 0.01E-3
const Ttot = 1.0
const taus_hawk = [3.0,3.0];

# Let's plot the activation function

theplot = let vs = range(-90,50.,length=150)
  rs = @. iofunction(vs,v0,α)
  plot(vs,rs;linewidth=2,leg=false,xlabel="voltage (mV)",ylabel="rate (Hz)",color=:black)
end;
plot(theplot)

## #src

# I run the rate model by Euler integration, with timesteps of 0.01 ms (the variable `dt`)

function euler_dv(v::Vector{R},_weights::Matrix{R},_input::Vector{R},
    _taus::Vector{R},_v0::R,_α::R) where R<:Real
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
end;

## #src

runtimes,runrates = run_2D_network();

# rates and times have been stored in the variables above!

# ## Hawkes process

# To define the Hawkes process, I need to find equivalent rates and currents
# that account for the input-output function that I used.

weights_equiv = @. abs(α^2*weights)
inputs_equiv =  α .* (input .- (α .* weights+I)*fill(v0,2))
ps_e,tr_e = H.population_state_exp_and_trace(1,taus_hawk[1])
ps_i,tr_i = H.population_state_exp_and_trace_inhibitory(1,taus_hawk[2])

# define conenctions
conn_ee = H.ConnectionExpKernel(onedmat(weights_equiv[1,1]),tr_e)
conn_ie = H.ConnectionExpKernel(onedmat(weights_equiv[2,1]),tr_e)
conn_ei = H.ConnectionExpKernel(onedmat(weights_equiv[1,2]),tr_i)
conn_ii = H.ConnectionExpKernel(onedmat(weights_equiv[2,2]),tr_i);

# define populations

pop_e = H.PopulationExpKernel(ps_e,inputs_equiv[1:1],(conn_ee,ps_e),(conn_ei,ps_i) )
pop_i = H.PopulationExpKernel(ps_i,inputs_equiv[2:2],(conn_ie,ps_e),(conn_ii,ps_i) );

# define recorder:  
# I record all spiketimes, although I only need the rate

nrec = 10_010
rec = H.RecFullTrain(nrec,2);

# define network:

netw = H.RecurrentNetworkExpKernel((pop_e,pop_i),(rec,));

# run network

function run_simulation!(network,num_spikes)
  t_now = 0.0
  H.reset!(network) # clear spike trains etc
  for _ in 1:num_spikes
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end

nspikes = nrec - 1
t_end=run_simulation!(netw,nspikes)

## #src
recc = H.get_content(rec) 
rates_e = H.numerical_rates(recc,1,t_end;pop_idx=1)
rates_i = H.numerical_rates(recc,1,t_end;pop_idx=2)

# ## Result: compare the rate in the two sytems

@info "final rates are $(round.(runrates[:,end];digits=2))"
@info "Hawkes final rates are $(round.(first.((rates_e,rates_i));digits=2))"

# **THE END**

## publish in documentation #src
thisfile = joinpath(splitpath(@__FILE__)[end-1:end]...) #src
using Literate; Literate.markdown(thisfile,"docs/src";documenter=true,repo_root_url="https://github.com/dylanfesta/HawkesSimulator.jl/blob/master") #src