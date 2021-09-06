using ProgressMeter
using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors ; theme(:dark) ; plotlyjs()
using SparseArrays 
using BenchmarkTools
using FFTW
using Random
Random.seed!(0)

push!(LOAD_PATH, abspath(@__DIR__,".."))

using  HawkesSimulator; const global H = HawkesSimulator

function onedmat(x::Real)
  ret=Matrix{Float64}(undef,1,1)
  ret[1,1] = x 
  return ret
end 

##

function onedmat(x::R) where R
  ret=Matrix{R}(undef,1,1)
  ret[1,1] = x 
  return ret
end;

mytau = 0.5  # kernel time constant
myw = onedmat(0.85) # weight: this needs to be a matrix
myinput = [0.7,] # this needs to be a vector
pop = H.PopulationExp(mytau);

function doplot() # Julia likes functions
  ts = range(-1.0,5;length=150)
  y = map(t->H.interaction_kernel(t,pop) , ts )
  plot(ts , y ; linewidth=3,leg=false,xlabel="time (s)",
      ylabel="interaction kernel")
end

doplot()

pops = H.PopulationState(pop,myinput)
ntw = H.InputNetwork(pops,[pops,],[myw,]); 


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

mytrain = pops.trains_history[1]
mydt = 0.1
myτmax = 25.0
mytaus = H.get_times(mydt,myτmax)
ntaus = length(mytaus)
cov_num = H.covariance_self_numerical(mytrain,mydt,myτmax);

##

mybinslong = 0.0:mydt:mytrain[end]-mydt
binned=fit(Histogram,mytrain,mybinslong;closed=:left).weights

ttot = mytrain[end]
nbinn = sum(binned)

myrate = nbinn/ttot

sum(binned.^2)/(ttot*mydt) - myrate^2

count(binned .> 1)/count(binned .== 1) 

##

cov_num[1]


ratefou / mydt

ratefou - (ratefou^2*mydt)

ratefou*(inv(mydt) - ratefou)

##
# Now I compute the covariance density analytically, at higher resolution,
# and I compare the two.

function four_high_res(dt::Real,Tmax::Real) # higher time resolution, longer time
  k1,k2 = 2 , 0.08
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

function four_high_res2(dt::Real,Tmax::Real) # higher time resolution, longer time
  k1,k2 = 3 , 0.001
  myτmax = Tmax * k1
  dt *= k2
  mytaus = H.get_times(dt,myτmax)
  nkeep = div(length(mytaus),k1)
  myfreq = H.get_frequencies_centerzero(dt,myτmax)
  gfou = myw[1,1] .* H.interaction_kernel_fourier.(myfreq,Ref(pop)) |> ifftshift
  ffou = let r=ratefou
    covf(g) = r /((1-g)*(1-g'))
    map(covf,gfou)
  end
  retf = real.(ifft(ffou)) ./ dt
  retf[1] *= dt # first element is rate!
  return mytaus[1:nkeep],retf[1:nkeep]
end


(taush,covfou)=four_high_res(mydt,myτmax)
(taush2,covfou2)=four_high_res2(mydt,myτmax)


##

covfou2[1]


##

function doplot()
  plt = plot(xlabel="time delay (s)",ylabel="Covariance density")
  plot!(plt,mytaus[2:end], cov_num[2:end]; linewidth=3, label="simulation" )
  plot!(plt,taush[1:end],covfou[1:end]; label="analytic",linewidth=3,linestyle=:dash)
  plot!(plt,taush2[2:end],covfou2[2:end]; label="analytic2",linewidth=3,linestyle=:dash)
  return plt
end
doplot()
