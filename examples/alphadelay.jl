
# # 1D Hawkes process with delayed alpha kernel

#=
In this example, I simulate either a 1D or a 2D Hawkes process, with a
delayed-alpha interaction kernel

```math
g(t) = H(t-t_{\text{delay}}) \,  \frac{(t-t_{\text{delay}})}{\tau^2} \,
 \exp\left(- \frac{(t-t_{\text{delay}})}{\tau} \right)
```

where $H(x)$ is the Heaviside function: $H(x)=0$ for $x<0$, $H(x)=1$ for $t\geq 0$.

Kernels are always normalized so that their integral is 1.
=#

# ## Initialization
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:default) #; plotlyjs()
using FFTW

using ProgressMeter
using Random
Random.seed!(1) # zero does not look as nice :-P

using HawkesSimulator; const global H = HawkesSimulator

function onedmat(x::Real)
  ret=Matrix{Float64}(undef,1,1)
  ret[1,1] = x 
  return ret
end 

##
# ## Define and visualize the kernel

mytau = 0.5
mydelay = 2.0
myw = onedmat(0.85)
myinput = [0.5,]

pop = H.PopulationAlphaDelay(mytau,mydelay)

function doplot() # Julia likes functions
  ts = range(-0.5,8;length=150)
  y = [H.interaction_kernel(_t,pop) for _t in ts]
  plot(ts,y ; linewidth=3,leg=false,xlabel="time (s)",
      ylabel="interaction kernel")
end

doplot()

# Note that the kernel starts well after zero.

# As a side note: in order to simulate Hawkes processes, one always 
# needs to define a *non-increasing upper limit* to the kernel. 
# This is what it looks like for this kernel. 

function doplot() # Julia likes functions
  ts = range(-0.5,8;length=150)
  y = [H.interaction_kernel(_t,pop) for _t in ts]
  yu = [H.interaction_kernel_upper(_t,pop) for _t in ts]
  plt = plot(ts,y ; linewidth=3,xlabel="time (s)",
      ylabel="interaction kernel", label="true kernel")
  plot!(plt, ts,yu ; linewidth=2, label="upper limit", linestyle=:dash)   
  return plt    
end

doplot()

# when defining a new kernel in the code, 
# the function `interaction_kernel(...)` 
# one needs also to define a non-increasing upper limit function,
# `interaction_kernel_upper(...)`
# the closer it is to the true kernel, the more efficient the simulation.

##
# ## Build the network and run it
pops = H.PopulationState(pop,myinput)
ntw = H.InputNetwork(pops,[pops,],[myw,]) 

function run_simulation!(netw,nspikes)
  t_now = 0.0
  H.reset!(netw) # clear spike trains etc
  @showprogress 1.0 "Running Hawkes process..." for k in 1:nspikes
    t_now = H.dynamics_step!(t_now,[netw,])
    if k%1_000 == 0
      H.clear_trains!(netw.postpops,50.0) # longer time horizon, because of delay 
    end
  end
  H.clear_trains!(netw.postpops,-1.0);
  return t_now
end

n_spikes = 80_000
Tmax = run_simulation!(ntw,n_spikes);

ratenum = H.numerical_rates(pops)[1]

ratefou = let gfou0 = myw[1,1] * H.interaction_kernel_fourier(0,pop)
  myinput[1]/(1-real(gfou0)) 
end

@info "Mean rate -  numerical $(round(ratenum;digits=2)), Fourier  $(round(ratefou;digits=2))"

##
# ## Covariance density

# First, compute it numerically for a reasonable time step
mytrain = pops.trains_history[1]
mydt = 0.1
myτmax = 60.0
mytaus = H.get_times(mydt,myτmax)
ntaus = length(mytaus)
cov_num = H.covariance_self_numerical(mytrain,mydt,myτmax);

# now compute it analytically, at higher resolution, and compare the two

function four_high_res(dt::Real,Tmax::Real) # higher time resolution, longer time
  k1,k2 = 2 , 0.1
  myτmax = Tmax * k1
  dt *= k2
  mytaus = H.get_times(dt,myτmax)
  nkeep = div(length(mytaus),k1)
  myfreq = H.get_frequencies_centerzero(dt,myτmax)
  gfou = myw[1,1] .* H.interaction_kernel_fourier.(myfreq,Ref(pop)) |> ifftshift
  ffou = let r=ratefou
    covf(g) = r*(g+g'-g*g')/((1-g)*(1-g'))
    map(covf,gfou)
  end
  retf = real.(ifft(ffou)) ./ dt
  return mytaus[1:nkeep],retf[1:nkeep]
end

taush,covfou=four_high_res(mydt,myτmax)

function doplot()
  plt = plot(xlabel="time delay (s)",ylabel="Covariance density")
  plot!(plt,mytaus[2:end], cov_num[2:end] ; linewidth=3, label="numerical" )
  return plot!(plt,taush[1:end],covfou[1:end]; label="analytic",linewidth=3,linestyle=:dash)
end

doplot()

# **THE END**

# Literate.markdown("examples/alphadelay.jl","docs/src";documenter=true,repo_root_url="https://github.com/dylanfesta/HawkesSimulator.jl/blob/master") #src