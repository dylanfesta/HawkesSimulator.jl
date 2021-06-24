
# # 2D process with delayed-alpha mutual interactions and exp self interaction

#=
Here I consider two neurons with a delayed interaction,
that self-interact without delay.
=#

# ## Initialization

using ProgressMeter
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:default)
using SparseArrays 
using FFTW
using Random
Random.seed!(0)

using HawkesProcesses; global const H = HawkesProcesses

##

# ## Define the interaction kernel
# The kernel is a delayed-alpha, see that example file for definition and
# details.
#
# Important : the autaptic weights (diagonal of `wmat`) are set to zero
# if not, there would also be a self-delayed-interaction. 

mytau = 0.6
mydelay = 1.0
mywmat = [ 0.0   0.1 
           1.8   0.0 ]

myinput = [0.2,0.1]

pop = H.PopulationAlphaDelay(mytau,mydelay)
pops = H.PopulationState(pop,myinput)

# ## Define the autapses
# the autaptic weights are `mywautaps`
# the self-interaction kernel is `popautaps`

mywautaps = [0.5,0.2]
popautaps = H.PopulationExp(0.4)
autapses_pop = H.Autapses(mywautaps,popautaps)

# ## Build the network and run simulation

ntw = H.InputNetwork(pops,[pops,],[mywmat,],autapses_pop) 

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

n_spikes = 100_000
Tmax = run_simulation!(ntw,n_spikes);

@info "Total duration $(round(Tmax;digits=1))"

# ## Compare rate , simulation vs analytic

ratenum = H.numerical_rates(pops)

# analytic rate from fourier Eq between 6 and  7 in Hawkes
# note that I need to add autapses to `G0` separately

ratefou = let G0 =  mywmat .* H.interaction_kernel_fourier(0,pop)
  G0 .+=  diagm( 0 => mywautaps .* H.interaction_kernel_fourier(0,popautaps) )
  inv(I-G0)*myinput|> real
end 

@info "Mean rate -  simulation $(round.(ratenum;digits=2)), analytic  $(round.(ratefou;digits=2))"

##
# ## Covariance density
# As in previous examples, compute from simulaton and then analytic

myspikes_all = pops.trains_history

mydt = 0.1
myτmax = 30.0
mytaus = H.get_times(mydt,myτmax)
ntaus = length(mytaus)
cov_num = H.covariance_density_numerical(myspikes_all,mydt,myτmax)

# plot all numerical covariance densities

function doplot()
  plt = plot(xlabel="time delay (s)",ylabel="Covariance density",title="simulation")
  for i in 1:2, j in 1:2
    plot!(plt,mytaus[2:end-1],cov_num[i,j,2:end-1] ; linewidth = 2,label="cov $i - $j")
  end
  return plt
end
doplot()

# the bumps reflect the delayed interactions

function four_high_res(dt::Real,Tmax::Real) 
  k1 = 2
  k2 = 0.2
  myτmax,mydt = Tmax * k1, dt*k2
  mytaus = H.get_times(mydt,myτmax)
  nkeep = div(length(mytaus),k1)
  myfreq = H.get_frequencies_centerzero(mydt,myτmax)
  G_omega = map(mywmat) do w
    ifftshift( w .* H.interaction_kernel_fourier.(myfreq,Ref(pop)))
  end
  for k in 1:size(mywmat,1) # add autapses
    waut = mywautaps[k]
    Gaut = ifftshift(waut .*  H.interaction_kernel_fourier.(myfreq,Ref(popautaps)))
    G_omega[k,k] .+= Gaut
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


(taush,Cfou)=four_high_res(mydt,myτmax)


function oneplot(i,j)
  plt = plot(xlabel="time delay (s)",ylabel="Covariance density",title="cov $i - $j")
  plot!(plt,mytaus[2:end],cov_num[i,j,2:end] ; linewidth = 3)
  return plot!(plt,taush[2:end],Cfou[i,j,2:end]; linestyle=:dash, linewidth=3)
end

# 1
oneplot(1,1)
# 2
oneplot(1,2)
# 3 
oneplot(2,1)
# 4
oneplot(2,2)

# Seems that there is an unpleasant boundary artifact for the first few timesteps.
# There is also a bit of overshooting, I have no idea why.

##
# ## Visualize raster plot 
# the raster plot shows some correlation between the neural activities
# neuron one (lower row) excites neuron two (upper row)

function doplot(tlims = (3000.,3080.) )
  plt=plot()
  for (is,_train) in enumerate(myspikes_all)
    train = filter(t-> tlims[1]< t < tlims[2],_train)
    nspk = length(train)
    scatter!(plt,train,fill(0.1*(is),nspk),markersize=30,
      markercolor=:black,markershape=:vline,leg=false)
  end
  return plot!(plt,ylims=(0.0,0.3))
end

doplot()

# **THE END**

# Literate.markdown("examples/2d_delay_autapses.jl","docs/src";documenter=true,repo_root_url="https://github.com/dylanfesta/HawkesProcesses.jl/blob/master") #src