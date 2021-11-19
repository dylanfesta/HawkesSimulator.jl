# # 1D and 2D Hawkes processes with exponential kernel

#=
In this example, I simulate first a 1D self-exciting Hawkes process and then a 2D one.
The interaction kernel is an exponentially decaying function, defined as:

```math
g(t) = H(t) \,  \frac{1}{\tau} \, \exp\left(- \frac{t}{\tau} \right)
```

where $H(t)$ is the Heaviside function: zero for $t<0$, one for $t\geq 0$.

Note that Kernels are always normalized so that their integral 
between $-\infty$ and $+\infty$ is 1.
=#

# ## Initialization
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:default) ; gr()
using FFTW

using ProgressMeter
using Random
Random.seed!(0)

using HawkesSimulator; const global H = HawkesSimulator

"""
    onedmat(x::R) where R
Generates a 1-by-1 Matrix{R} that contains `x` as only element   
"""
function onedmat(x::R) where R
  return cat(x;dims=2)
end;

## #src
# First I define the kernel, and the self-interaction weight.
# The kernel is defined through a "Population": all neurons in the same 
# population have the same kernel.
#
# `myw` is a scaling factor (the weight of the 
# autaptic connection). The baseline rate is given by `myinput`

mytau = 0.5  # kernel time constant
myw = onedmat(0.85) # weight: this needs to be a matrix
myinput = [0.7,] # this needs to be a vector
mykernel = H.KernelExp(mytau);

# This is the plot of the (self) interaction kernel
# (before  the scaling by `myw`)

theplot = let ts = range(-1.0,5;length=150),
  y = map(t->H.interaction_kernel(t,mykernel) , ts )
  plot(ts , y ; linewidth=3,leg=false,xlabel="time (s)",
     ylabel="interaction kernel")
end;
plot(theplot)

# Now I build the network, using the simplified constructor

popstate = H.PopulationState(mykernel,1)
ntw = H.RecurrentNetwork(popstate,myw,myinput);

## #src
# ## Simulation
# The length of the simulation is measured by the total number of spikes
# here called `num_spikes`.  
# The function `flush_trains!(...)` is used to store older spikes as 
# history and let them be ignored by the kernels.
# The time parameters should be regulated based on the kernel shape.

function run_simulation!(network,num_spikes,
    t_flush_trigger=300.0,t_flush=100.0)
  t_now = 0.0
  H.reset!(network) # clear spike trains etc
  @showprogress "Running Hawkes process..." for _ in 1:num_spikes
    t_now = H.dynamics_step_singlepopulation!(t_now,network)
    H.flush_trains!(network,t_flush_trigger;Tflush=t_flush)
  end
  H.flush_trains!(network) # flush everything into history
  return t_now
end

n_spikes = 100_000
Tmax = run_simulation!(ntw,n_spikes)
ratenum = H.numerical_rates(popstate;Tend=Tmax)[1]
@info "Simulation completed, mean rate $(round(ratenum;digits=2)) Hz"

##
# ## Visualize raster plot of the events

# the raster plot shows some correlation between the neural activities
# neuron one (lower row) excites neuron two (upper row)

function rasterplot(tlims = (1100.,1120.) )
  _train = popstate.trains_history[1]
  plt=plot()
  train = filter(t-> tlims[1]< t < tlims[2],_train)
  nspk = length(train)
  scatter!(plt,train,fill(0.1,nspk),markersize=30,
      markercolor=:black,markershape=:vline,leg=false)
  return plot!(plt,ylims=(0.0,0.2),xlabel="time (s)")
end

rasterplot()

# event times are always stored in `pops.trains_history`

##  #src
# ## Plot the instantaneous rate

#=
This is the probability of a spike given the past activity up until that 
moment. It is usually denoted by $\lambda^*(t)$.
=#

function get_insta_rate(t,popstate)
  _train = popstate.trains_history[1]
  myinput[1] + H.interaction(t,_train,myw[1],popstate.unittype)
end
function plot_instarate(popstate,tlims=(1100,1120))
  tplot = range(tlims...;length=100) 
  _train = popstate.trains_history[1]
  tspk = filter(t-> tlims[1]<=t<=tlims[2],_train) # add the exact spiketimes for cleaner plot
  tplot = sort(vcat(tplot,tspk,tspk .- 1E-4))
  plt=plot(xlabel="time (s)",ylabel="instantaneous rate")
  plot!(plt,t->get_insta_rate(t,popstate),tplot;linewidth=2,color=:black)
  scatter!(t->get_insta_rate(t,popstate),tspk;leg=false)
  avg_rate = H.numerical_rates(popstate;Tend=Tmax)[1]
  ylim2 = ylims()[2]
  plot!(plt,tplot, fill(avg_rate,length(tplot)),
     color=:red,linestyle=:dash,ylims=(0,ylim2))
end
plot_instarate(popstate)

# ## Plot the total event counts

# Total number of events as a function of time. It grows linearly, and the
# steepness is pretty much the rate. 

function plot_counts(tlims=(0.,1000.))
  avg_rate = H.numerical_rates(popstate;Tend=Tmax)[1]
  tplot = range(tlims...;length=100)
  _train = popstate.trains_history[1]
  nevents(tnow::Real) = count(t-> t <= tnow,_train)
  plt=plot(xlabel="time (s)",ylabel="number of events",leg=false)
  plot!(plt,tplot,nevents.(tplot),color=:black,linewidth=2)
  plot!(plt,tplot , tplot .* avg_rate,color=:red,linestyle=:dash)
end
plot_counts()

# ## Rate
#=
Now I compare the numerical rate with the analytic solution.

The analytic rate corresponds to the stationary solution  
of a linear dynamical system (assumung all stationary rates are above zero).
```math
\frac{\mathrm d \mathbf r}{\mathrm d t} = - \mathbf r + W\;r + \mathbf h \quad;
\qquad  r_\infty = (I-W)^{-1} \; \mathbf h
```
=#

rate_analytic = inv(I-myw)*myinput
rate_analytic = rate_analytic[1] # 1-D , just a scalar

@info "Mean rate -  numerical $(round(ratenum;digits=2)), analytic  $(round(rate_analytic;digits=2))"

##
# ## Covariance density

# TODO : write definition
#
# I compute the covariance density it numerically.
# The time inteval `mydt` should not be too small.

mytrain = popstate.trains_history[1]
mydt = 0.1
myτmax = 25.0
mytaus = H.get_times(mydt,myτmax)
ntaus = length(mytaus)
cov_num = H.covariance_self_numerical(mytrain,mydt,myτmax);

##
# Now I compute the covariance density analytically, at higher resolution,
# and I compare the two.

function four_high_res(dt::Real,Tmax::Real) # higher time resolution, longer time
  k1,k2 = 2 , 0.01
  myτmax = Tmax * k1
  dt *= k2
  mytaus = H.get_times(dt,myτmax)
  nkeep = div(length(mytaus),k1)
  myfreq = H.get_frequencies_centerzero(dt,myτmax)
  gfou = myw[1,1] .* H.interaction_kernel_fourier.(myfreq,Ref(popstate)) |> ifftshift
  ffou = let r=rate_analytic
    covf(g) = r/((1-g)*(1-g'))
    map(covf,gfou)
  end
  retf = real.(ifft(ffou)) ./ dt
  retf[1] *= dt  # first element is rate
  return mytaus[1:nkeep],retf[1:nkeep]
end

(taush,covfou)=four_high_res(mydt,myτmax)

function doplot()
  plt = plot(xlabel="time delay (s)",ylabel="Covariance density")
  plot!(plt,mytaus[2:end], cov_num[2:end] ; linewidth=3, label="simulation" )
  plot!(plt,taush[2:end],covfou[2:end]; label="analytic",linewidth=3,linestyle=:dash)
  return plt
end
doplot()

# 1D system completed !
##
# ## Same results, but in a 2D system

myτ = 1/2.33
mywmat = [ 0.31   0.3 
           0.9  0.15 ]
myin = [1.0,0.1]           
p1 = H.KernelExp(myτ)
ps1 = H.PopulationState(p1,2)
ntw = H.RecurrentNetwork(ps1,mywmat,myin)

# ## Start the simulation
# the function `run_simulation!(...)` has been defined above
# Note that `n_spikes` is the total number of spikes
# among **all** units in the system.

n_spikes = 500_000 

Tmax = run_simulation!(ntw,n_spikes,100.0,10.0);

# ## Check the rates
# The analytic rate is from  Eq between 6 and  7 in Hawkes 1971

num_rates = H.numerical_rates(ps1)
myspikes_both = ps1.trains_history

ratefou = let G0 =  mywmat .* H.interaction_kernel_fourier(0,p1)
  inv(I-G0)*myin |> real
end 
rate_analytic = inv(I-mywmat)*myin 

@info "Total duration $(round(Tmax;digits=1)) s"
@info "Rates are $(round.(num_rates;digits=2))"
@info "Analytic rates are $(round.(rate_analytic;digits=2)) Hz"

##    #src
# ## Covariance density
# there are 4 combinations, therefore I will compare 4 lines.

mydt = 0.1
myτmax = 15.0
mytaus = H.get_times(mydt,myτmax)
ntaus = length(mytaus)
cov_num = H.covariance_density_numerical(myspikes_both,mydt,myτmax)

function doplot()
  plt=plot(xlabel="time delay (s)",ylabel="Covariance density")
  for i in 1:2, j in 1:2
    plot!(plt,mytaus[2:end-1],cov_num[i,j,2:end-1], linewidth = 3, label="cov $i-$j")
  end
  return plt
end

doplot()

# The analytic solution is eq 12 from Hawkes 1971 

function four_high_res(dt::Real,Tmax::Real) 
  k1 = 2
  k2 = 0.005
  myτmax,mydt = Tmax * k1, dt*k2
  mytaus = H.get_times(mydt,myτmax)
  nkeep = div(length(mytaus),k1)
  myfreq = H.get_frequencies_centerzero(mydt,myτmax)
  G_omega = map(mywmat) do w
    ifftshift( w .* H.interaction_kernel_fourier.(myfreq,Ref(p1)))
  end
  D = Diagonal(ratefou)
  M = Array{ComplexF64}(undef,2,2,length(myfreq))
  Mt = similar(M,Float64)
  for i in eachindex(myfreq)
    G = getindex.(G_omega,i)
    M[:,:,i] = (I-G)\D/(I-G') 
  end
  for i in 1:2,j in 1:2
    Mt[i,j,:] = real.(ifft(M[i,j,:]))
    Mt[i,j,2:end] ./= mydt # diagonal of t=0 contains the rate
  end
  return mytaus[1:nkeep],Mt[:,:,1:nkeep]
end

taush,Cfou=four_high_res(mydt,myτmax)

function oneplot(i,j)
  plt=plot(xlabel="time delay (s)",ylabel="Covariance density",title="cov $i - $j")
  plot!(plt,mytaus[2:end],cov_num[i,j,2:end] ; linewidth = 3, label="simulation")
  plot!(plt,taush[2:end],Cfou[i,j,2:end]; linestyle=:dash, linewidth=3, label="analytic")
end

# 1
oneplot(1,1)

# 2
oneplot(1,2)

# 3
oneplot(2,1)

# 4
oneplot(2,2)


# **THE END**

# Literate.markdown("examples/exp_1and2D.jl","docs/src";documenter=true,repo_root_url="https://github.com/dylanfesta/HawkesSimulator.jl/blob/master") #src