
# # 1D and 2D Hawkes processes with exponential kernel

#=
In this example, I simulate either a 1D or a 2D Hawkes process, with a negative exponential interaction kernel

```math
g(t) = H(t) \,  \frac{1}{\tau} \, \exp\left(- \frac{t}{\tau} \right)
```

where $H(t)$ is the Heaviside function: zero for $t<0$, one for $t\geq 0$.

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

"""
    onedmat(x::R) where R
Generates a Matrix{R} with the specified element inside of it  
"""
function onedmat(x::R) where R
  ret=Matrix{R}(undef,1,1)
  ret[1,1] = x 
  return ret
end;

##

# First I define the kernel, and a self interaction weight
# The interaction kernel is defined by a corresponding population type.
# `mytau` is the parameter $\tau$. `myw` is a scaling factor (the weight of the 
# autaptic connection). The baseline rate is given by `myinput`

mytau = 0.5
myw = onedmat(0.85) # thie needs to be a matrix
myinput = [0.7,] # this needs to be a vector
pop = H.PopulationExp(mytau);

# This is the plot of the (self) interaction kernal
# (before it is scaled by `myw`)

function doplot() # Julia likes functions
  ts = range(-1.0,5;length=150)
  y = [H.interaction_kernel(_t,pop) for _t in ts]
  plot(ts,y ; linewidth=3,leg=false,xlabel="time (s)",
      ylabel="interaction kernel")
end

doplot()

# Now I build the network

pops = H.PopulationState(pop,myinput)
ntw = H.InputNetwork(pops,[pops,],[myw,]); 

# ## Simulation
# The length of the simulation is expressed as a total number of spikes
# here called `n_spikes`.  
# The function `clear_trains!(...)` is used to store older spikes as history and 
# let them be ignored by the kernels

function run_simulation!(netw,nspikes)
  t_now = 0.0
  H.reset!(netw) # clear spike trains etc
  @showprogress 1.0 "Running Hawkes process..." for k in 1:nspikes
    t_now = H.dynamics_step!(t_now,[netw,])
    if k%1_000 == 0
      H.clear_trains!(netw.postpops)
    end
  end
  H.clear_trains!(netw.postpops,-1.0);
  return t_now
end

n_spikes = 80_000
Tmax = run_simulation!(ntw,n_spikes);
##
# ## Visualize raster plot of the events

# the raster plot shows some correlation between the neural activities
# neuron one (lower row) excites neuron two (upper row)

function doplot(tlims = (3000.,3100.) )
  _train = pops.trains_history[1]
  plt=plot()
  train = filter(t-> tlims[1]< t < tlims[2],_train)
  nspk = length(train)
  scatter!(plt,train,fill(0.1,nspk),markersize=30,
      markercolor=:black,markershape=:vline,leg=false)
  return plot!(plt,ylims=(0.0,0.2),xlabel="time (s)")
end

doplot()

# event times are always stored in `pops.trains_history`

##
# ## Rate
# Now I compare the numerical rate with the analytic solution, that used the Fourier transform 

ratenum = H.numerical_rates(pops)[1]

# this is the general equation for rate in a self-excting Hawkes process, from Hawkes 1976
ratefou = let gfou0 = myw[1,1] * H.interaction_kernel_fourier(0,pop)
  myinput[1]/(1-real(gfou0)) 
end;

@info "Total duration $(round(Tmax;digits=1)) s"
@info "Mean rate -  numerical $(round(ratenum;digits=2)), analytic  $(round(ratefou;digits=2))"

##
# ## Covariance density

# Now the covariance density
# first, compute it numerically for a reasonable time range

mytrain = pops.trains_history[1]
mydt = 0.1
myτmax = 25.0
mytaus = H.get_times(mydt,myτmax)
ntaus = length(mytaus)
cov_num = H.covariance_self_numerical(mytrain,mydt,myτmax);

##
# now compute it analytically, at higher resolution
# and compare the two.

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

(taush,covfou)=four_high_res(mydt,myτmax)

function doplot()
  plt = plot(xlabel="time delay (s)",ylabel="Covariance density")
  plot!(plt,mytaus[2:end], cov_num[2:end] ; linewidth=3, label="simulation" )
  plot!(plt,taush,covfou; label="analytic",linewidth=3,linestyle=:dash)
  return plt
end
doplot()

# 1D system completed !
##
# ## Initialize the system in 2D

myτ = 1/2.33
mywmat = [ 0.31   0.3 
           0.9  0.15 ]
myin = [1.0,0.1]           
p1 = H.PopulationExp(myτ)
ps1 = H.PopulationState(p1,myin)
ntw = H.InputNetwork(ps1,[ps1,],[mywmat,]);

# ## Start the simulation
# the function `run_simulation!(...)` has been defined above
# Note that `n_spikes` is the total between **all** units in the system.

n_spikes = 500_000 

Tmax = run_simulation!(ntw,n_spikes);

# ## Check the rates
# The analytic rate is from  Eq between 6 and  7 in Hawkes 1976

num_rates = H.numerical_rates(ps1)
myspikes_both = ps1.trains_history

ratefou = let G0 =  mywmat .* H.interaction_kernel_fourier(0,p1)
  inv(I-G0)*myin |> real
end 

@info "Total duration $(round(Tmax;digits=1)) s"
@info "Rates are $(round.(num_rates;digits=2))"
@info "Analytic rates are $(round.(ratefou;digits=2)) Hz"

##
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

# The analytic solution is eq 12 from Hawkes 1976 

function four_high_res(dt::Real,Tmax::Real) 
  k1 = 2
  k2 = 0.2
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
    M[:,:,i] = (I-G)\D*(G+G'-G*G')/(I-G') 
  end
  for i in 1:2,j in 1:2
    Mt[i,j,:] = real.(ifft(M[i,j,:])) ./ mydt
  end
  return mytaus[1:nkeep],Mt[:,:,1:nkeep]
end

taush,Cfou=four_high_res(mydt,myτmax)

function oneplot(i,j)
  plt=plot(xlabel="time delay (s)",ylabel="Covariance density",title="cov $i - $j")
  plot!(plt,mytaus[2:end],cov_num[i,j,2:end] ; linewidth = 3, label="simulation")
  plot!(plt,taush,Cfou[i,j,:]; linestyle=:dash, linewidth=3, label="analytic")
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