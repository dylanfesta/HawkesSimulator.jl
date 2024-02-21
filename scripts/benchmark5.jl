using LinearAlgebra,Statistics
push!(LOAD_PATH, abspath(@__DIR__,".."))
using Profile,PProf,BenchmarkTools
using Random ; Random.seed!(0)

using HawkesSimulator ; global  const H = HawkesSimulator

##


const τexc = 2.0
const τinh = 1.0
const hall = 1.0
const αquad = 0.3
const w_full_good = [0 1.0 -0.5;
                      0.5 0 -0.3;
                      0.25 0.3 0]
const w_full_bad = [0 2.0 -0.5;
                    0.5 0 -0.3;
                    0.25 0.3 0]
const ne,ni = 2,1

const hvect = fill(hall,ne+ni)


function run_simulation!(network,num_spikes)
  t_now = 0.0
  H.reset!(network) # clear spike trains etc
  for _ in 1:num_spikes
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end

function test_with_Hawkes_quadratic_norecorder(ne::Integer,ni::Integer,
    W::Matrix{R},h::Vector{R},αgain::R,
    γ::Real;
    τe::Real=0.5,τi::Real=0.3) where R

  ntot = ne+ni
  idxe = 1:ne
  idxi = ne+1:ntot
  nspikes = round(Int,γ*1E6)
  @info "Running simulation with $nspikes spikes"
  @assert size(W,1) == size(W,2) == length(h) == ntot
  @assert all(h .>= 0)
  @assert all(>=(zero(R)), W[:,1:ne])
  @assert all(<=(zero(R)), W[:,ne+1:end])

  Wabs = abs.(W)*sqrt(αgain)
  hscal = sqrt(αgain)*h
  ps_e,tr_e = H.population_state_exp_and_trace(ne,τe)
  ps_i,tr_i = H.population_state_exp_and_trace_inhibitory(ni,τi)

  conn_ee = H.ConnectionExpKernel(Wabs[idxe,idxe],tr_e)
  conn_ie = H.ConnectionExpKernel(Wabs[idxi,idxe],tr_e)
  conn_ei = H.ConnectionExpKernel(Wabs[idxe,idxi],tr_i)
  conn_ii = H.ConnectionExpKernel(Wabs[idxi,idxi],tr_i)

  nl = H.NLRectifiedQuadratic()
  pop_e = H.PopulationExpKernel(ps_e,hscal[idxe],(conn_ee,ps_e),(conn_ei,ps_i),nonlinearity=nl)
  pop_i = H.PopulationExpKernel(ps_i,hscal[idxi],(conn_ie,ps_e),(conn_ii,ps_i),nonlinearity=nl)
  
  netw = H.RecurrentNetworkExpKernel((pop_e,pop_i))

  t_end=run_simulation!(netw,nspikes)
  
  return t_end
end

# Quick run to precompile

t_test = test_with_Hawkes_quadratic_norecorder(ne,ni,w_full_good,hvect,αquad,0.1;
  τe=τexc,τi=τinh)

##

@time test_with_Hawkes_quadratic_norecorder(ne,ni,w_full_good,hvect,αquad,1.;
  τe=τexc,τi=τinh)

println("Test with good weights completed\n Now running test with bad weights!")

@time test_with_Hawkes_quadratic_norecorder(ne,ni,w_full_bad,hvect,αquad,1.;
  τe=τexc,τi=τinh)

println("Test with bad weights completed. Now bad, but three times as long.")

@time test_with_Hawkes_quadratic_norecorder(ne,ni,w_full_bad,hvect,αquad,3.0;
  τe=τexc,τi=τinh)

##


