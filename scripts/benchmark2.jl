using LinearAlgebra,Statistics
using InvertedIndices # for Not()
using IterTools # for product()
push!(LOAD_PATH, abspath(@__DIR__,".."))
using Profile,PProf,BenchmarkTools
using Random ; Random.seed!(0)

using HawkesSimulator ; global  const H = HawkesSimulator

##

const γ = 0.2:0.1:2.0
const θ = 0.2:0.1:2.3
const gammatheta = collect(product(γ,θ))[:]
const n_iters = length(gammatheta)
const nthreads = Threads.nthreads()
@info "Using $nthreads threads"


# this function runs the network dynamics
# it simply runs one-spike iteration steps for the given number of spikes
# returns the time at the end of the simulation
function run_simulation!(network,num_spikes;t_start::Float64=0.0)
  t_now = t_start
  # H.reset!(network) # clear spike trains etc
  for _ in 1:num_spikes
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end

function mysim(γval::Real,θval::Real,cnt::Int64;γspikes::Real=1.5)
    Random.seed!(cnt)
    # network parameters
    ne = 3
    τe = 10E-3
    he = 5.0 
    w_min = 1E-5
    w_max = 1/(ne-1) - 1E-5
    # set random weights
    w_mat = w_min.+(w_max-w_min).*rand(Float64, (ne,ne))
    # set zero diagonal
    w_mat[diagind(w_mat)] .= 0.0
    w_0 = copy(w_mat)
    # consider a saturated w_mat as worse case scenario
    w_mat_max = fill(w_max,ne,ne)
    w_mat_max[diagind(w_mat_max)] .= 0.0

    # expected neural rates (analytic) at the beginning, and in case of saturation.
    he_vec = fill(he,ne)
    ## create a population state and a trace
    # (the trace is necessary for the connection)
    pse,trae = H.population_state_exp_and_trace(ne,τe;label="population_e")

    # the population needs to be connected to something, but before
    # creating a connectin, we need to define its plasticity rule

    plasticity_ee = let A = 1E-6,  # change A to a negative value, to get an anti-Hebbian rule
        τplus = 40E-3,
        Aplus = A/τplus,
        τminus = γval*τplus,
        Aminus = -θval*A/τminus
        bounds=H.PlasticityBoundsLowHigh(w_min,w_max)
        H.PairSTDP(τplus,τminus,Aplus,Aminus,ne,ne;
            plasticity_bounds=bounds)
    end
    connection_ee = H.ConnectionExpKernel(w_mat,trae,plasticity_ee)
    #connection_ee = H.ConnectionExpKernel(w_mat,trae)

    pop_e = H.PopulationExpKernel(pse,he_vec,(connection_ee,pse))

    n_spikes = Int(γspikes*1E7)

    # now we can finally build the network. A network is a collection of populations and a
    # collection of recorders.
    the_network = H.RecurrentNetworkExpKernel((pop_e,))
    t_end = run_simulation!(the_network,n_spikes)

    return t_end
end

mutable struct Atomic{T}; @atomic x::T; end
    
function time_me_please(trial)
  niter = Atomic(n_iters)
  output = Dict()
  Threads.@threads for k in 1:n_iters
    (γval,θval) = gammatheta[k]
    output["a_$(k)"] = mysim(γval,θval,k+10*trial*n_iters)
    @atomic niter.x -= 1 
    println("*********")
    println("remaining iterations: $(niter.x)")
  end
  trying = Dict("saved" => output)
  return trying
end

##
# run function one for precompilation
@time mysim(0.1,0.1,1;γspikes=0.1)
@time mysim(0.1,0.1,1;γspikes=2.0)
@time mysim(0.1,0.1,1;γspikes=4.0)
@profview mysim(0.1,0.1,1;γspikes=4.0)

## allocation profiler!
# Collect an allocation profile
Profile.Allocs.clear()
Profile.Allocs.@profile mysim(0.1,0.1,1)
# Export pprof allocation profile and open interactive profiling web interface.
PProf.Allocs.pprof()

## normal profiling
@profview mysim(0.1,0.1,1)


##
for trial = 1:10
    @time output1 = time_me_please(trial)
    matwrite("Hrnd_$(trial).mat", output1)
end

##

plasticity_ee = let A = 1E-6,  # change A to a negative value, to get an anti-Hebbian rule
    τplus = 40E-3,
    Aplus = A/τplus,
    τminus = τplus,
    Aminus = -A/τminus
    bounds=H.PlasticityBoundsLowHigh(1E-5,10.)
    H.PairSTDP(τplus,τminus,Aplus,Aminus,50,50;
        plasticity_bounds=bounds)
end

using Cthulhu

descend(plasticity_ee.bounds,Tuple{Float64,Float64}) 

@descend plasticity_ee.bounds(1.0, plasticity_ee.traceminus.val[3])
@code_warntype plasticity_ee.bounds(1.0, plasticity_ee.traceminus.val[3])