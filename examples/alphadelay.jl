
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
Random.seed!(0)

using HawkesSimulator; const global H = HawkesSimulator

##
# ## Define and visualize the kernel

mytau = 0.5
mydelay = 2.0
myw = fill(0.85,(1,1)) # 1x1 matrix
myinput = [0.5,]       # 1-dim vector

ker = H.KernelAlphaDelay(mytau,mydelay)

plt = let ts = range(-0.5,8;length=150)
  y = [H.interaction_kernel(_t,ker) for _t in ts]
  plt=plot(ts,y ; linewidth=3,leg=false,xlabel="time (s)",
      ylabel="interaction kernel")
  ymax=ylims()[2]    
  plot!(plt,[0,0],[0,ymax];linestyle=:dash,linecolor=:black)
end;
plot(plt)

# Note that the kernel starts after zero, according to the delay indicated.

# As a side note: in order to simulate Hawkes processes, one always 
# needs to define a *non-increasing upper limit* to the kernel. 
# This is what it looks like for this kernel. 

plt = let  ts = range(-0.5,8;length=150)
  y = [H.interaction_kernel(_t,ker) for _t in ts]
  yu = [H.interaction_kernel_upper(_t,ker) for _t in ts]
  plt = plot(ts,y ; linewidth=3,xlabel="time (s)",
      ylabel="interaction kernel", label="true kernel")
  plot!(plt, ts,yu ; linewidth=2, label="upper limit", linestyle=:dash)   
  ymax=ylims()[2]    
  plot!(plt,[0,0],[0,ymax];linestyle=:dash,linecolor=:black,label="")
end;
plot(plt)

# the closer the upper limit is to the true kernel,
# the more efficient the simulation.

##      #src
# ## Build the network and run it
# I compare the final rate with what I expect from the analytic solution
# (see first example file)

pops = H.PopulationState(ker,1)
ntw = H.RecurrentNetwork(pops,myw,myinput)

function run_simulation!(network,num_spikes,
    t_flush_trigger=300.0,t_flush=100.0)
  t_now = 0.0
  H.reset!(network) # clear spike trains etc
  @showprogress "Running Hawkes process..." for _ in 1:num_spikes
    t_now = H.dynamics_step!(t_now,network)
    H.flush_trains!(network,t_flush_trigger;Tflush=t_flush)
  end
  H.flush_trains!(network) # flush everything into history
  return t_now
end

n_spikes = 80_000
Tmax = run_simulation!(ntw,n_spikes);

ratenum = H.numerical_rates(pops)[1]
rate_analytic = (I-myw)\myinput
rate_analytic = rate_analytic[1] # from 1-dim vector to scalar

@info "Mean rate -  numerical $(round(ratenum;digits=2)), analytic  $(round(rate_analytic;digits=2))"

##
# ## Covariance density

# First, compute covariance density numerically for a reasonable time step
mytrain = pops.trains_history[1]
mydt = 0.1
myτmax = 60.0
mytaus = H.get_times(mydt,myτmax)
ntaus = length(mytaus)
cov_num = H.covariance_self_numerical(mytrain,mydt,myτmax);

# now compute covariance density
#  analytically (as in Hawkes models), at higher resolution, 
# and compare analytic and numeric
# 
# Note that the high resolution is not just for a better plot, but also to 
# ensure the result is more precise when we move from frequency domain 
# (Fourier transforms) to time domain.

function four_high_res(dt::Real,Tmax::Real) # higher time resolution, longer time
  k1,k2 = 2 , 0.01
  myτmax = Tmax * k1
  dt *= k2
  mytaus = H.get_times(dt,myτmax)
  nkeep = div(length(mytaus),k1)
  myfreq = H.get_frequencies_centerzero(dt,myτmax)
  gfou = myw[1,1] .* H.interaction_kernel_fourier.(myfreq,Ref(ker)) |> ifftshift
  ffou = let r=rate_analytic
    covf(g) = r/((1-g)*(1-g'))
    map(covf,gfou)
  end
  retf = real.(ifft(ffou))
  retf[2:end] ./= dt # first element is rate
  return mytaus[1:nkeep],retf[1:nkeep]
end

taush,covfou=four_high_res(mydt,myτmax)

plt= let plt = plot(xlabel="time delay (s)",ylabel="Covariance density")
  plot!(plt,mytaus[2:end], cov_num[2:end] ; linewidth=3, label="numerical" )
  plot!(plt,taush[2:end],covfou[2:end]; label="analytic",linewidth=3,linestyle=:dash)
end;

plot(plt)

# Analytics and numerics match quite well.

# **THE END**

# Literate.markdown("examples/alphadelay.jl","docs/src";documenter=true,repo_root_url="https://github.com/dylanfesta/HawkesSimulator.jl/blob/master") #src