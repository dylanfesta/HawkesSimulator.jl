using ProgressMeter
using Test
using LinearAlgebra,Statistics,StatsBase,Distributions
using QuadGK
using Plots,NamedColors ; theme(:dark) ; plotlyjs()
using FileIO
using SparseArrays 
using BenchmarkTools
using FFTW
using Random
Random.seed!(0)

push!(LOAD_PATH, abspath(@__DIR__,".."))

using  HawkesSimulator; const global H = HawkesSimulator

##


# Vs 2D linear model 


##


##


##
num_rates = H.numerical_rates(popstate)
myspikes_both = popstate.trains_history


ratefou = let G0 =  w .* H.interaction_kernel_fourier(1E-3,population)
  inv(I-G0)*baseline_rate |> real
end

@info "Total duration $(round(Tmax;digits=1)) s"
@info "Rates are $(round.(num_rates;digits=2))"
@info "Analytic rates are $(round.(ratefou;digits=2)) Hz"

##

tmp_img = H.draw_spike_raster(network.postpops.trains_history,0.1,80.0;
  Tstart = 60.0,
  spike_colors=[colorant"blue",colorant"red"],
  spike_separator=15,
  spike_size = 20,
  background_color=RGB(1.,.9,.7))
save("/tmp/tmp.png",tmp_img) 

tmp_img = H.draw_spike_raster(repeat(network.postpops.trains_history;outer=3),0.1,80.0;
  Tstart = 60.0,
  spike_colors=repeat([colorant"blue",colorant"red"];outer=3),
  spike_separator=15,
  spike_size = 30,
  background_color=RGB(1.,.9,.7))
save("/tmp/tmp.png",tmp_img) 


##

num_neurons = 2
tau_l = 0.5
time_step = 0.001
t_max = 20
num_steps = Int(t_max/time_step)
lrate = zeros(num_steps, 2)
time = zeros(num_steps)
dr = zeros(2)

function drdt(j::Int64, r::Vector{Float64})
    d = h - r[j] + dot(w[j,:],r)
    return d
end

for i in 2:num_steps
    time[i] = time[i-1] + time_step
    for j in 1:num_neurons
        dr[j] = time_step*(drdt(j, lrate[i-1, :]))/tau_l
        lrate[i,j] = lrate[i-1, j] + dr[j]
    end
end

function plot_linear()
    plt = plot(xlabel="time (s)", ylabel="rate")
    plot!(plt, time , lrate, label = ["rate_E" "rate_I"],linewidth=2)
end

plot_linear()

##
print("linear rate of E = ",lrate[num_steps, 1])
print("linear rate of I = ",lrate[num_steps, 2])

##

# Hawkes
#  ri(t) = -K*Si + sum(Wij(K*Sj)) + hi

# ## Creating network

##

function interaction(t::R,train::Vector{R},w::R,prepop::H.UnitType,upperbound::Bool) where R
  if (iszero(w) || isempty(train))
    return zero(R)
  elseif upperbound
    return  w * mapreduce(tk -> H.interaction_kernel_upper(t-tk,prepop),+,train)
  else
    ret = w * mapreduce(tk -> H.interaction_kernel(t-tk,prepop),+,train)
    return ret
  end
end
function interaction(t::R,weights_in::AbstractVector{<:R},prepop::H.PopulationState,
    upperbound::Bool) where R
  ret = 0.0
  for (j,w) in enumerate(weights_in)
    train = popstate.trains_history[j]
    ret += interaction(t,train,w,prepop.populationtype,upperbound)
  end
  return ret
end
function insta_rate(t_now::Real,inp::H.InputNetwork,ineu::Integer;upperbound::Bool=false)
  ret = 0.0
  for (w,prepop) in zip(inp.weights,inp.prepops)
    w_in = view(w,ineu,:)
    ret += interaction(t_now,w_in,prepop,upperbound)
  end
  return ret
end

## Excitation Function vs Time

function  plot_excitation(trainsh, tlims=(0,10))
    global expected_lambda = H.numerical_rate.(trainsh)
    num_neurons = size(trainsh,1)
    times = range(tlims...;length=100)
    for i in 1:num_neurons
        trains = filter(t-> tlims[1]<=t<=tlims[2],trainsh[i])
        times = sort(vcat(times, trains))
    end
    plt = plot(xlabel="time (s)", ylabel="excitation function")
    y_vector = Array{Float64}(undef, length(times), num_neurons)
    expected_lambda_vector = zeros(Float64, length(times), num_neurons)
    for i in 1:num_neurons
        y_vector[:,i] = map(t->insta_rate(t,network,i;upperbound=false), times)
        y_vector[:,i] = y_vector[:,i] .+ baseline_rate[i]
        expected_lambda_vector[:,i] = expected_lambda_vector[:,i] .+ expected_lambda[i]
    end
    plot!(plt, times , y_vector, label = ["lambda1(t)" "lambda2(t)"],linewidth=2)
    plot!(plt, times , expected_lambda_vector, 
      label = ["E[lambda1(t)]" "E[lambda2(t)]"],linewidth=2)
end

plot_excitation(popstate.trains_history)


##

test1 = rand(100)
deleteat!(test1,(1:5))

test2 = rand(200)

boh = [test1;test2]

