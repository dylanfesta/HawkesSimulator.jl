

###### mean rates and other measures

function numerical_rates(ps::PopulationState;
    Tstart::Real=0.0,Tend::Real=Inf)
  return numerical_rate.(ps.trains_history;Tstart=Tstart,Tend=Tend)
end
function numerical_rates(pop::AbstractPopulation;
    Tstart::Real=0.0,Tend::Real=Inf)
  return numerical_rates(pop.state;Tstart=Tstart,Tend=Tend)
end

function numerical_rate(train::Vector{Float64};
    Tstart::Real=0.0,Tend::Real=Inf)
  isempty(train) && return 0.0  
  if !isfinite(Tend)
    Tend = train[end]
  end
  Δt = Tend - Tstart
  return length(train)/Δt
end

##########################
## covariance density

"""
    bin_spikes(Y::Vector{R},dt::R,Tend::R;Tstart::R=0.0) where R

# Arguments
  + `Y::Vector{<:Real}` : vector of spike times
  + `dt::Real` : time bin size
  + `Tend::Real` : end time for the raster
# Optional argument
  + `Tstart::Real=0.0` : start time for the raster

# Returns   
  + `binned_spikes::Vector{<:Integer}` : `binned_spikes[k]` is the number of spikes that occur 
      in the timebin `k`  (i.e. between `Tstart + (k-1)*dt` and `Tstart + k*dt`)
"""
function bin_spikes(Y::Vector{R},dt::R,Tend::R;Tstart::R=0.0) where R
  times = range(Tstart,Tend;step=dt)  
  ret = fill(0,length(times)-1)
  for y in Y
    if Tstart < y <= last(times)
      k = searchsortedfirst(times,y)-1
      ret[k] += 1
    end
  end
  return ret
end


function bin_and_rates(Y::Vector{R},dt::R,Tend::R;Tstart::R=0.0) where R
  times = range(Tstart,Tend;step=dt)  
  return midpoints(times),bin_spikes(Y,dt,Tend;Tstart=Tstart) ./ dt
end

function bin_and_rates(Ys::Vector{Vector{R}},dt::R,Tend::R;Tstart::R=0.0) where R
  times = range(Tstart,Tend;step=dt)
  ret = [bin_spikes(Y,dt,Tend;Tstart=Tstart) ./ dt for Y in Ys]
  return midpoints(times),vcat(transpose.(ret)...)
end


@inline function get_times_strict(dt::R,Tend::R;Tstart::R=0.0) where R<:Real
  return range(Tstart,Tend-dt;step=dt)
end


# and returns a value in Hz
function instantaneous_rates(idxs_neu::AbstractVector{<:Integer},
    dt::Float64,pop::AbstractPopulation;Tend::Float64=-1.0,Tstart::Float64=0.0)
  trains = pop.state.trains_history[idxs_neu]  
  return  instantaneous_rates(dt,trains;
   Tend=Tend,Tstart=Tstart)
end
function instantaneous_rates(dt::Float64,trains::Vector{Vector{Float64}};
  Tend::Float64=-1.0,Tstart::Float64=0.0)
  if Tend < 0 
    Tend = minimum(last.(trains))
  end
  tmid = midpoints(range(Tstart,Tend;step=dt))
  counts = fill(0.0,length(tmid))
  for train in trains
    counts .+= bin_spikes(train,dt,Tend;Tstart=Tstart)
  end
  return tmid,counts ./ (dt*length(trains))
end

# time starts at 0, ends at T-dt, there are T/dt steps in total
@inline function get_times(dt::Real,T::Real)
  return (0.0:dt:(T-dt))
end


# the first element (zero lag) is always rate/dτ
function covariance_self_numerical(Y::Vector{R},dτ::R,τmax::R,
     Tend::Union{R,Nothing}=nothing) where R
  τtimes,ret = covariance_density_numerical([Y,],dτ,τmax;verbose=false,Tend=Tend)
  return  τtimes, ret[:,1,1]
end


function covariance_density_numerical(Ys::Vector{Vector{R}},dτ::Real,τmax::R;
   Tend::Union{R,Nothing}=nothing,verbose::Bool=false) where R
  Tend = something(Tend, maximum(last,Ys)- dτ)
  ndt = round(Integer,τmax/dτ)
  n = length(Ys)
  ret = Array{Float64}(undef,ndt,n,n)
  if verbose
      @info "The full dynamical iteration has $(round(Integer,Tend/dτ)) bins ! (too many?)"
  end
  for i in 1:n
    binnedi = bin_spikes(Ys[i],dτ,Tend)
    fmi = length(Ys[i]) / Tend # mean frequency
    ndt_tot = length(binnedi)
    _ret_alloc = Vector{R}(undef,ndt)
    for j in 1:n
      if verbose 
        @info "now computing cov for pair $i,$j"
      end
      binnedj =  i==j ? binnedi : bin_spikes(Ys[j],dτ,Tend)
      fmj = length(Ys[j]) / Tend # mean frequency
      binnedj_sh = similar(binnedj)
      @inbounds @simd for k in 0:ndt-1
        circshift!(binnedj_sh,binnedj,k)
        _ret_alloc[k+1] = dot(binnedi,binnedj_sh)
      end
      @. _ret_alloc = (_ret_alloc / (ndt_tot*dτ^2)) - fmi*fmj
      ret[:,i,j] = _ret_alloc
    end
  end
  return get_times_strict(dτ,τmax), ret
end

function covariance_density_ij(Ys::Vector{Vector{R}},i::Integer,j::Integer,dτ::R,τmax::R;
   Tend::Union{R,Nothing}=nothing) where R
  return covariance_density_ij(Ys[i],Ys[j],dτ,τmax;Tend=Tend)
end

function covariance_density_ij(X::Vector{R},Y::Vector{R},dτ::Real,τmax::R;
    Tend::Union{R,Nothing}=nothing) where R
  times = get_times_strict(dτ,τmax)
  ndt = length(times)
  times_ret = vcat(-reverse(times[2:end]),times)
  Tend = something(Tend, max(X[end],Y[end])- dτ)
  ret = Vector{Float64}(undef,2*ndt-1)
  binnedx = bin_spikes(X,dτ,Tend)
  binnedy = bin_spikes(Y,dτ,Tend)
  fx = length(X) / Tend # mean frequency
  fy = length(Y) / Tend # mean frequency
  ndt_tot = length(binnedx)
  binned_sh = similar(binnedx)
  # 0 and forward
  @simd for k in 0:ndt-1
    circshift!(binned_sh,binnedy,-k)
    ret[ndt-1+k+1] = dot(binnedx,binned_sh)
  end
  # backward
  @simd for k in 1:ndt-1
    circshift!(binned_sh,binnedy,k)
    ret[ndt-k] = dot(binnedx,binned_sh)
  end
  @. ret = (ret / (ndt_tot*dτ^2)) - fx*fy
  return times_ret, ret
end



# frequencies for Fourier transform.
# from -1/dt to 1/dt - 1/T in steps of 1/T
function get_frequencies_centerzero(dt::Real,T::Real)
  dω = inv(T)
  ωmax = 0.5/dt
  f = dω:dω:ωmax
  ret = vcat(-reverse(f),0.,f[1:end-1])
  return ret
end

function get_frequencies(dt::Real,T::Real)
  dω = inv(T)
  ωmax = inv(dt)
  ret = 0:dω:ωmax-dω
  return ret
end


"""
  draw_spike_raster(trains::Vector{Vector{Float64}},
      dt::Real,Tend::Real;
      Tstart::Real=0.0,
      spike_size::Integer = 5,
      spike_separator::Integer = 1,
      background_color::Color=RGB(1.,1.,1.),
      spike_colors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),
      max_size::Real=1E4) where C<:Color

Draws a matrix that contains the raster plot of the spike train.

# Arguments
  + `Trains` :  Vector of spike trains. The order of the vector corresponds to 
    the order of the plot. First element is at the top, second is second row, etc.
  + `dt` : time interval representing one horizontal pixel  
  + `Tend` : final time to be considered

# Optional arguments
  + `Tstart::Real` : starting time
  + `spike_size::Integer` : heigh of spike (in pixels)
  + `spike_separator::Integer` : space between spikes, and vertical padding
  + `background_color::Color` : self-explanatory
  + `spike_colors::Union{Color,Vector{Color}}` : if a single color, color of all spikes, if vector of colors, 
     color for each neuron (length should be same as number of neurons)
  + `max_size::Integer` : throws an error if image is larger than this number (in pixels)

# Returns
  + `raster_matrix::Matrix{Color}` you can save it as a png file
"""
function draw_spike_raster(trains::Vector{Vector{Float64}},
  dt::Real,Tend::Real;
    Tstart::Real=0.0,
    spike_size::Integer = 5,
    spike_separator::Integer = 1,
    background_color::Color=RGB(1.,1.,1.),
    spike_colors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),
    max_size::Real=1E4) where C<:Color
  nneus = length(trains)
  if typeof(spike_colors) <: Color
    spike_colors = repeat([spike_colors,];outer=nneus)
  else
    @assert length(spike_colors) == nneus "error in setting colors"
  end
  binned_binary  = map(trains) do train
    .! iszero.(bin_spikes(train,dt,Tend;Tstart=Tstart))
  end
  ntimes = length(binned_binary[1])
  ret = fill(background_color,
    (nneus*spike_size + # spike sizes
      spike_separator*nneus + # spike separators (incl. top/bottom padding) 
      spike_separator),ntimes)
  @assert all(size(ret) .< max_size ) "The image is too big! Please change time limits"  
  for (neu,binv,col) in zip(1:nneus,binned_binary,spike_colors)
    spk_idx = findall(binv)
    _idx_pre = (neu-1)*(spike_size+spike_separator)+spike_separator
    y_color = _idx_pre+1:_idx_pre+spike_size
    ret[y_color,spk_idx] .= col
  end
  return ret
end

#=
Example of raster plot with Makie

f = let f= Figure()
  dt_rast=0.0005
  Δt_rast = 0.50
  Tstart_rast = 100.
  Tend_rast = Tstart_rast + Δt_rast
  rast_spk = 10
  rast_sep = 3
  theraster=H.draw_spike_raster(trains,dt_rast,Tend_rast;Tstart=Tstart_rast,
  spike_size=rast_spk,spike_separator=rast_sep)
  rast = permutedims(theraster)
  neu_yvals  = let n_neu = n_default,
    N = 2*n_neu*(rast_sep+rast_spk)+2*rast_sep+rast_spk
    ret = collect(range(0,n_neu+1;length=N))
    ret[rast_spk+1:end-rast_spk]
  end
  pxtimes,pxneus = size(rast)
  times = range(0,Δt_rast,length=pxtimes)
  ax1 = Axis(f[1, 1], aspect=pxtimes/pxneus,
    rightspinevisible=false,topspinevisible=false,
    xlabel="time (s)",ylabel="neuron #")
  image!(ax1,times,neu_yvals,rast)
  f
end
=#


# do draw spikes on existing raster plot
function add_to_spike_raster!(raster::Matrix,
    trains::Vector{Vector{Float64}},neu_idxs::Vector{Int64},
    dt::Real,Tend::Real, spike_color::C;
      Tstart::Real=0.0,
      spike_size::Integer = 5,
      spike_separator::Integer = 1,
      ) where C<:Color
  @assert length(neu_idxs) == length(trains)    
  binned_binary  = map(trains) do train
    .! iszero.(bin_spikes(train,dt,Tend;Tstart=Tstart))
  end
  ntimes = length(binned_binary[1])
  @assert size(raster,2) == ntimes "Mismatch in time binning!"
  for (neu,binv) in zip(neu_idxs,binned_binary)
    spk_idx = findall(binv)
    _idx_pre = (neu-1)*(spike_size+spike_separator)+spike_separator
    y_color = _idx_pre+1:_idx_pre+spike_size
    raster[y_color,spk_idx] .= spike_color
  end
  return nothing
end


#### this is the part which involves reader objects

function get_spiketimes_spikeneurons(rec::Union{RecFullTrain,RecFullTrainContent},
    (pop_idx::Integer)=1)
  spiketimes,spikeneurons = rec.timesneurons[pop_idx]
  spkt = filter(isfinite,spiketimes)
  spkn = filter(>(0),spikeneurons)
  return spkt,spkn
end
function spiketn_to_trains(spiketimes::Vector{<:Real},
    spikeneurons::Vector{<:Integer},Nneus::Integer)
  trains = map(_-> Float64[],1:Nneus)
  for (spkt,spkn) in zip(spiketimes,spikeneurons)
    push!(trains[spkn],spkt)
  end
  return trains
end
function get_trains(rec::Union{RecFullTrain,RecFullTrainContent},Nneus::Integer,
    (pop_idx::Integer)=1)
  spiketn = get_spiketimes_spikeneurons(rec,pop_idx)
  return spiketn_to_trains(spiketn...,Nneus)
end

function numerical_rates(rec::Union{RecFullTrain,RecFullTrainContent},Nneus::Integer,Tend::Real;
    pop_idx::Integer=1,Tstart::Real=0.0)
  @assert Tstart < Tend "Tstart must be smaller than Tend, got $Tstart and $Tend instead"  
  trains = get_trains(rec,Nneus,pop_idx)  
  return numerical_rate.(trains;Tstart=Tstart,Tend=Tend)
end

function binned_spikecount(trains::Vector{Vector{R}},dt::Float64,Tend::Real;
      neurons_idx::AbstractVector=Int64[],
      Tstart::Float64=0.0) where R
  if !isempty(neurons_idx)
    trains = trains[neurons_idx]
  end
  Nneus = length(trains)
  tbins = Tstart:dt:Tend
  ntimes = length(tbins)-1
  binnedcount = fill(0,(Nneus,ntimes))
  for (neu,train) in enumerate(trains)
    for t in train
      if (tbins[1] < t <= tbins[end]) # just in case
        tidx = searchsortedfirst(tbins,t)-1
        binnedcount[neu,tidx]+=1
      end
    end
  end
  binsc=midpoints(tbins)
  return binsc,binnedcount
end


function instantaneous_rates(trains::Vector{Vector{R}},dt::R,Tend::Real;
    neurons_idx::AbstractArray=Int64[],
    Tstart::Float64=0.0) where R
  @assert Tend > Tstart+dt "Tend should be greater than Tstart+dt, but got Tend=$Tend and Tstart+dt=$(Tstart+dt)"  
  binsc,counts = binned_spikecount(trains,dt,Tend;neurons_idx=neurons_idx,Tstart=Tstart)  
  return binsc, (counts./dt)
end


