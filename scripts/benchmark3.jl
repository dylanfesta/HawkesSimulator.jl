using LinearAlgebra,Statistics
using InvertedIndices # for Not()
using IterTools # for product()
push!(LOAD_PATH, abspath(@__DIR__,".."))
using Profile,PProf,BenchmarkTools
using Random ; Random.seed!(0)

using HawkesSimulator ; global  const H = HawkesSimulator

##

function run_simulation!(network,num_spikes)
  t_now = 0.0
  H.reset!(network) # clear spike trains etc
  for _ in 1:num_spikes
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end



# the parameter γ decides the number of spikes

function test_with_Hawkes_2pop_norecorder(γ::Real,W::Matrix{R},h::Vector{R};
    τe::Real=0.5,τi::Real=0.3) where R

  nspikes = round(Int,γ*1E6)
  @assert size(W,1) == size(W,2) == length(h) == 2
  @assert all(h .>= 0)
  @assert all(W[:,1] .>= 0)
  @assert all(W[:,2] .<= 0)

  Wabs = abs.(W)
  ne,ni = 1,1
  ps_e,tr_e = H.population_state_exp_and_trace(ne,τe)
  ps_i,tr_i = H.population_state_exp_and_trace_inhibitory(ni,τi)
  onedmat(x::Real) = cat(x;dims=2)

  conn_ee = H.ConnectionExpKernel(onedmat(Wabs[1,1]),tr_e)  
  conn_ie = H.ConnectionExpKernel(onedmat(Wabs[2,1]),tr_e)
  conn_ei = H.ConnectionExpKernel(onedmat(Wabs[1,2]),tr_i)
  conn_ii = H.ConnectionExpKernel(onedmat(Wabs[2,2]),tr_i)

  #nl = H.NLRectifiedQuadratic()
  #nl = H.NLRmax(300.0)
  #pop_e = H.PopulationExpKernel(ps_e,h[1:1],(conn_ee,ps_e),(conn_ei,ps_i),nonlinearity=nl)
  #pop_i = H.PopulationExpKernel(ps_i,h[2:2],(conn_ie,ps_e),(conn_ii,ps_i),nonlinearity=nl)
  
  pop_e = H.PopulationExpKernel(ps_e,h[1:1],(conn_ee,ps_e),(conn_ei,ps_i))
  pop_i = H.PopulationExpKernel(ps_i,h[2:2],(conn_ie,ps_e),(conn_ii,ps_i))
  #nrec = 10_000_000
  #rec = H.RecFullTrain(nrec,2)
  netw = H.RecurrentNetworkExpKernel((pop_e,pop_i))

  t_end=run_simulation!(netw,nspikes)
  
  # recc = H.get_content(rec)
  # rates_e = H.numerical_rates(recc,1,t_end;pop_idx=1,Tstart=60.0)[1]
  # rates_i = H.numerical_rates(recc,1,t_end;pop_idx=2,Tstart=60.0)[1]
  # #@info "Hawkes final rates are $(round.((rates_e,rates_i);digits=2))"
  #return (t_end=t_end,rec_trains=recc,
  #  final_rates=[rates_e,rates_i])
  return t_end
end


function test_with_Hawkes_2pop_quadratic_norecorder(γ::Real,W::Matrix{R},h::Vector{R};
    τe::Real=0.5,τi::Real=0.3) where R

  nspikes = round(Int,γ*1E6)
  @assert size(W,1) == size(W,2) == length(h) == 2
  @assert all(h .>= 0)
  @assert all(W[:,1] .>= 0)
  @assert all(W[:,2] .<= 0)

  Wabs = abs.(W)
  ne,ni = 1,1
  ps_e,tr_e = H.population_state_exp_and_trace(ne,τe)
  ps_i,tr_i = H.population_state_exp_and_trace_inhibitory(ni,τi)
  onedmat(x::Real) = cat(x;dims=2)

  conn_ee = H.ConnectionExpKernel(onedmat(Wabs[1,1]),tr_e)  
  conn_ie = H.ConnectionExpKernel(onedmat(Wabs[2,1]),tr_e)
  conn_ei = H.ConnectionExpKernel(onedmat(Wabs[1,2]),tr_i)
  conn_ii = H.ConnectionExpKernel(onedmat(Wabs[2,2]),tr_i)

  nl = H.NLRectifiedQuadratic()
  #nl = H.NLRmax(300.0)
  pop_e = H.PopulationExpKernel(ps_e,h[1:1],(conn_ee,ps_e),(conn_ei,ps_i),nonlinearity=nl)
  pop_i = H.PopulationExpKernel(ps_i,h[2:2],(conn_ie,ps_e),(conn_ii,ps_i),nonlinearity=nl)
  
  netw = H.RecurrentNetworkExpKernel((pop_e,pop_i))

  t_end=run_simulation!(netw,nspikes)
  
  # recc = H.get_content(rec)
  # rates_e = H.numerical_rates(recc,1,t_end;pop_idx=1,Tstart=60.0)[1]
  # rates_i = H.numerical_rates(recc,1,t_end;pop_idx=2,Tstart=60.0)[1]
  # #@info "Hawkes final rates are $(round.((rates_e,rates_i);digits=2))"
  #return (t_end=t_end,rec_trains=recc,
  #  final_rates=[rates_e,rates_i])
  return t_end
end

const htest = [5.0,5.0]
const wtest = 0.001 .* [1.275 -0.65;
                1.1 -0.5] 
const τtest = [200E-3, 100E-3]

# run once to precompile
test_with_Hawkes_2pop_norecorder(0.001,wtest,htest;τe=τtest[1],τi=τtest[2])
test_with_Hawkes_2pop_quadratic_norecorder(0.001,wtest,htest;τe=τtest[1],τi=τtest[2])

##

@time test_with_Hawkes_2pop_norecorder(1.0,wtest,htest;τe=τtest[1],τi=τtest[2])
@time test_with_Hawkes_2pop_norecorder(10.0,wtest,htest;τe=τtest[1],τi=τtest[2])

##

@profview test_with_Hawkes_2pop_norecorder(10.0,wtest,htest;τe=τtest[1],τi=τtest[2])

##

@time test_with_Hawkes_2pop_quadratic_norecorder(1.0,wtest,htest;τe=τtest[1],τi=τtest[2])
@time test_with_Hawkes_2pop_quadratic_norecorder(10.0,wtest,htest;τe=τtest[1],τi=τtest[2])

##

@profview test_with_Hawkes_2pop_quadratic_norecorder(10.0,wtest,htest;τe=τtest[1],τi=τtest[2])


