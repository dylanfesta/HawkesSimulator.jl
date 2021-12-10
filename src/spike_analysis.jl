

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
  Tend = min(Tend,train[end])
  Δt = Tend - Tstart
  return length(train)/Δt
end

##########################
## covariance density

function bin_spikes(Y::Vector{R},dt::R,Tend::R;
    Tstart::R=0.0) where R
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

function covariance_self_numerical(Y::Vector{R},dτ::R,τmax::R,
     Tmax::Union{R,Nothing}=nothing) where R
  ret = covariance_density_numerical([Y,],dτ,τmax,Tmax;verbose=false)
  return ret[1,1,:]
end

function covariance_density_numerical(Ys::Vector{Vector{R}},dτ::Real,τmax::R,
   Tmax::Union{R,Nothing}=nothing ; verbose::Bool=false) where R
  Tmax = something(Tmax, maximum(x->x[end],Ys)- dτ)
  ndt = round(Integer,τmax/dτ)
  n = length(Ys)
  ret = Array{Float64}(undef,n,n,ndt)
  if verbose
      @info "The full dynamical iteration has $(round(Integer,Tmax/dτ)) bins ! (too many?)"
  end
  for i in 1:n
    binnedi = bin_spikes(Ys[i],dτ,Tmax)
    fmi = length(Ys[i]) / Tmax # mean frequency
    ndt_tot = length(binnedi)
    _ret_alloc = Vector{R}(undef,ndt)
    for j in 1:n
      if verbose 
        @info "now computing cov for pair $i,$j"
      end
      binnedj =  i==j ? binnedi : bin_spikes(Ys[j],dτ,Tmax)
      fmj = length(Ys[j]) / Tmax # mean frequency
      binnedj_sh = similar(binnedj)
      @inbounds @simd for k in 0:ndt-1
        circshift!(binnedj_sh,binnedj,k)
        _ret_alloc[k+1] = dot(binnedi,binnedj_sh)
      end
      @. _ret_alloc = _ret_alloc / (ndt_tot*dτ^2) - fmi*fmj
      ret[i,j,:] = _ret_alloc
    end
  end
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
+ `Tstart` : starting time
+ `max_size` : throws an error if image is larger than this number (in pixels)
+ `spike_size` : heigh of spike (in pixels)
+ `spike_separator` : space between spikes, and vertical padding
+ `background_color` : self-explanatory
+ `spike_colors` : if a single color, color of all spikes, if vector of colors, 
color for each neuron (length should be same as number of neurons)

# returns
`raster_matrix::Matrix{Color}` you can save it as a png file

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


#### this is the part which involves reader objects

function get_spiketimes_spikeneurons(rec::RecFullTrain,
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
function get_trains(rec::RecFullTrain,Nneus::Integer,
    (pop_idx::Integer)=1)
  spiketn = get_spiketimes_spikeneurons(rec,pop_idx)
  return spiketn_to_trains(spiketn...,Nneus)
end

function numerical_rates(rec::RecFullTrain,Nneus::Integer,Tend::Real,
  (pop_idx::Integer)=1;Tstart=0.0)
  trains = get_trains(rec,Nneus,pop_idx)  
  return numerical_rate.(trains;Tstart=Tstart,Tend=Tend)
end

function binned_spikecount(trains::Vector{Vector{R}},dt::Float64,Tend::Real;
      neurons_idx::AbstractArray=Int64[],
      Tstart::Float64=0.0)
  if !isempty(neurons_idx)
    trains = trains[neurons_idx]
  end
  Nneus = length(trains)
  tbins = Tstart:dt:Tend
  ntimes = length(tbins)-1
  binnedcount = fill(0,(Nneus,ntimes))
  for (neu,train) in enumerate(trans)
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

function instantaneous_rates(trains::Vector{Vector{R}},dt::Float64,Tend::Real;
    neurons_idx::AbstractArray=Int64[],
    Tstart::Float64=0.0)
  binsc,counts = binned_spikecount(trains,dt,Tend;neurons_idx=neurons_idx,Tstart=Tstart)  
  return binsc, (counts./dt)
end