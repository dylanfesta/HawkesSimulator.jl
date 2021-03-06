using HawkesSimulator ; global const H = HawkesSimulator
using Test
using Distributions,Statistics,StatsBase,LinearAlgebra
using QuadGK
using Random ; Random.seed!(0)


function rates_analytic(W::Matrix{R},r0::Vector{R}) where R
  return (I-W)\r0
end
@testset "kernels" begin
  # integral is 1
  mykernels = let tau = rand(Uniform(0.3,0.8))
    k1 = H.KernelExp(tau)
    k2 = H.KernelStep(15.0)
    tau = rand(Uniform(0.3,0.8))
    taud = rand(Uniform(0.1,0.4))
    k3 = H.KernelAlphaDelay(tau,taud)
    (k1,k2,k3)
  end

  for ker in mykernels
    integr =  quadgk(t->H.interaction_kernel(t,ker),-3.0,20.0)[1]
    @test isapprox(integr,1;atol=0.001)
  end

  # to do - Fourier!!!

end

@testset "Flush spiketrain" begin
  hnew ,spktnew = H._flush_train!([1.,2.,3.],[ 4., 5., 12. , 13. ],3.0+3.0) 
  @test all(hnew .== [1.,2,3,4,5.])
  @test all(spktnew .== [12.,13.])
end


@testset "2D Vs linear model" begin
  weights = [0.31 -0.3
            0.9  -0.15]
  inputs = [5.0 , 5.0]
  rates_analytic  = inv(I-weights)*inputs

  tau =  0.5 
  mykernel = H.KernelExp(tau) #considering single population
  nneus = length(inputs)
  popstate = H.PopulationState(mykernel,nneus)
  network = H.RecurrentNetwork(popstate,weights,inputs)

  function simulate!(network,num_spikes)
    t_now = 0.0
    H.reset!(network) # clear spike trains etc
    for _ in 1:num_spikes
      t_now = H.dynamics_step!(t_now,network)
      H.flush_trains!(popstate,10.0;Tflush=5.0)
    end
    H.flush_trains!(popstate)
    return t_now
  end
  n_spikes = 200_000
  Tmax = simulate!(network,n_spikes)
  rates_ntw = H.numerical_rates(network.populations[1])
  @test all(isapprox.(rates_analytic,rates_ntw;rtol=0.2))

  # do it again, checking the one-population optimized function 
  function simulate_onepop!(network,num_spikes)
    t_now = 0.0
    H.reset!(network) # clear spike trains etc
    for _ in 1:num_spikes
      t_now = H.dynamics_step_singlepopulation!(t_now,network)
      H.flush_trains!(popstate,10.0;Tflush=5.0)
    end
    H.flush_trains!(popstate)
    return t_now
  end
  n_spikes = 200_000
  Tmax = simulate!(network,n_spikes)
  rates_ntw = H.numerical_rates(network.populations[1])
  @test all(isapprox.(rates_analytic,rates_ntw;rtol=0.2))

end


@testset "Warmup phase" begin
  N1 = 33
  N2 = 12
  weights1 = fill(Inf,N1,N1)
  inputs1 = fill(Inf,N1)
  weights2 = fill(Inf,N2,N2)
  inputs2 = fill(Inf,N2)

  tau =  123.5 
  mykernel1 = H.KernelExp(tau) #considering single population
  mykernel2 = H.KernelExp(rand()*tau) #considering single population

  popstate1 = H.PopulationState(mykernel1,N1)
  popstate2 = H.PopulationState(mykernel2,N2)

  population1 = H.PopulationHawkes(popstate1,inputs1,
    (H.ConnectionWeights(weights1),popstate1) )
  population2 = H.PopulationHawkes(popstate2,inputs2,
    (H.ConnectionWeights(weights2),popstate2) )

  # multi-pop constructor
  network = H.RecurrentNetwork(population1,population2)

  ##

  target_rates1 = rand(Uniform(5.,12.),N1)
  target_rates2 = rand(Uniform(12.,44.),N2)
  target_rates = [target_rates1,target_rates2]

  Ttot = 500.0

  H.do_warmup!(Ttot,network,target_rates)
  H.flush_trains!(network)

  @test all( isapprox.(H.numerical_rates(population1),target_rates1;rtol=0.2) )
  @test all( isapprox.(H.numerical_rates(population2),target_rates2;rtol=0.2) )
end

@testset "Poisson input units" begin
  N1 = 13
  N2 = 17
  target_rates1 = rand(Uniform(5.,12.),N1)
  target_rates2 = rand(Uniform(12.,44.),N2)

  population1 = let spikegen = H.SGPoisson(target_rates1)
    H.PopulationInput(spikegen,N1)
  end
  population2 = let spikegen = H.SGPoisson(target_rates2)
    H.PopulationInput(spikegen,N2)
  end

  network = H.RecurrentNetwork(population1,population2)
  function simulate!(network,num_spikes)
    t_now = 0.0
    H.reset!(network) # clear spike trains etc
    for _ in 1:num_spikes
      t_now = H.dynamics_step!(t_now,network)
    end
    H.flush_trains!(network)
    return t_now
  end

  n_spikes = 200_000
  Tmax = simulate!(network,n_spikes)
  rates_ntw1 = H.numerical_rates(network.populations[1])
  rates_ntw2 = H.numerical_rates(network.populations[2])

  @test all(isapprox.(target_rates1,rates_ntw1;rtol=0.1))
  @test all(isapprox.(target_rates2,rates_ntw2;rtol=0.1))
end

@testset "Input units, time-varying function" begin
  Nunits = 200
  function rate_fun(t::Float64,::Integer)
    return 10.0 + 6.0*cos(t*(2??)*0.5)
  end
  function rate_fun_upper(::Float64,::Integer)
    return 10.0 + 6.0
  end
  #plot(t->rate_fun(t,0),range(0,10;length=300);leg=false)
  population1 = let spikegen = H.SGPoissonFunction(rate_fun,rate_fun_upper)
    H.PopulationInput(spikegen,Nunits)
  end
  network = H.RecurrentNetwork(population1)
  function simulate!(network,num_spikes)
    t_now = 0.0
    H.reset!(network) # clear spike trains etc
    for _ in 1:num_spikes
      t_now = H.dynamics_step_singlepopulation!(t_now,network)
    end
    H.flush_trains!(network)
    return t_now
  end
  n_spikes = 50_000
  Tmax = simulate!(network,n_spikes)
  # now, compute rates!
  tmid,rats = H.instantaneous_rates(collect(1:Nunits),0.1,population1;Tend=Tmax)
  @test all(isapprox.(rats,rate_fun.(tmid,1);rtol=0.33))
end

@testset "Input units, given spike train" begin
  # post-pre spikes paradigm
  # (nor really needed here, but useful for plasticity)
  function ps_trains(rate::R,??t_ro::R,Ttot::R;
      tstart::R = 0.05) where R
    post = collect(range(tstart,Ttot; step=inv(rate)))
    pre = post .- ??t_ro
    return [pre,post] 
  end
  function get_test_pop(rate,nreps,??t_ro,connection_test)
    Ttot = nreps/rate
    prepostspikes = ps_trains(rate,??t_ro,Ttot) 
    gen = H.SGTrains(prepostspikes)
    state = H.PopulationState(H.InputUnit(gen),2)
    return H.PopulationInputTestWeights(state,connection_test)
  end
  connection_test = H.ConnectionWeights(fill(0.0,2,2))
  nreps = 100
  population = get_test_pop(0.5,nreps,2E-3,connection_test)
  population.state.unittype.spike_generator.trains
  network = H.RecurrentNetwork(population)
  function simulate!(network,num_spikes)
    t_now = 0.0
    H.reset!(network) # clear spike trains etc
    for _ in 1:num_spikes
      t_now = H.dynamics_step_singlepopulation!(t_now,network)
    end
    H.flush_trains!(network)
    return t_now
  end
  n_spikes = nreps*2-3
  Tmax = simulate!(network,n_spikes)
  trains_out1 =  population.state.trains_history[1]
  target_trains1 = population.state.unittype.spike_generator.trains[1]
  trains_out2 =  population.state.trains_history[2]
  target_trains2 = population.state.unittype.spike_generator.trains[2]
  ntest = nreps-10
  @test all(trains_out1[1:ntest] .== target_trains1[1:ntest] )
  @test all(trains_out2[1:ntest] .== target_trains2[1:ntest] )
end


@testset "special case of exponential kernel" begin
  nneus = 1
  tauker = 12.34E-2
  trace_ker = H.Trace(tauker,nneus,H.ForDynamics())
  H.update_for_dynamics!(trace_ker,1)
  get_trace(t) = H.trace_proposal!([NaN,],t,trace_ker)[1]
  _area = quadgk(get_trace,0,2.0)[1]
  @test isapprox(_area,1.0;atol=1E-5)

  # like 2D test above, but for ExpKernel types

  nneus = 2
  tauker = 0.5
  trace_ker = H.Trace(tauker,nneus,H.ForDynamics())
  trace_useless = H.Trace(123.0,nneus,H.ForPlasticity())

  popstate = H.PopulationStateExpKernel(nneus,trace_ker,trace_useless)
  myweights = [0.31 -0.3
              0.9  -0.15]
  myinputs = [5.0 , 5.0]
  rates_analytic  = inv(I-myweights)*myinputs

  connection = H.ConnectionExpKernel(myweights,trace_ker)
  population = H.PopulationExpKernel(popstate,connection,myinputs)

  n_spikes = 10_000
  recorder1 = H.RecFullTrain(n_spikes,1)
  recorder2 = H.RecFullTrain(n_spikes,1)
  network1 = H.RecurrentNetworkExpKernel(population,recorder1)
  network2 = H.RecurrentNetworkExpKernel(population,recorder2)

  function simulate1!(network,num_spikes)
    t_now = 0.0
    H.reset!(network) # clear spike trains etc
    for _ in 1:num_spikes
      t_now = H.dynamics_step!(t_now,network)
    end
    return t_now
  end
  function simulate2!(network,num_spikes)
    t_now = 0.0
    H.reset!(network) # clear spike trains etc
    for _ in 1:num_spikes
      t_now = H.dynamics_step_singlepopulation!(t_now,network)
    end
    return t_now
  end
  Tmax1 = simulate1!(network1,n_spikes)
  Tmax2 = simulate2!(network2,n_spikes)
  rates_ntw1 = H.numerical_rates(recorder1,nneus,Tmax1)
  rates_ntw2 = H.numerical_rates(recorder2,nneus,Tmax2)
  @test all(isapprox.(rates_analytic,rates_ntw1;rtol=0.2))
  @test all(isapprox.(rates_analytic,rates_ntw2;rtol=0.2))
end

@testset "Heterosynaptic plasticity initialization" begin
  n = 523
  theweights = rand(n,n)
  theweights[diagind(theweights)] .= 0.0
  # heterosynaptic plasticity
  ??t_het = 1929.0
  wmax = 1.0
  wmin = 0.03
  wsum_max =  19.0
  het_tol = 0.01
  het_constr = H.HetStrictSum(wsum_max,wmin,wmax,het_tol)
  het_met = H.HetAdditive()
  het_targ = H.HetBoth()
  het_plast = H.PlasticityHeterosynapticApprox(n,n,??t_het,het_constr,het_met,het_targ)
  H.plasticity_init_weights!(theweights,het_plast;repeats=100)
  @test all( isapprox.(sum(theweights;dims=1),wsum_max;atol=3*het_tol))
  @test all( isapprox.(sum(theweights;dims=2),wsum_max;atol=3*het_tol))
end

@testset "Exp kernel and forced spiking output" begin

  nneus = 3
  tauker = 1.5
  rates_start = 50.0
  Tend = 60.0

  noweights = zeros(nneus,nneus)
  noinputs = zeros(nneus)

  ## generate trains
  trains = [ H.make_poisson_samples(rates_start,Tend) for _ in 1:nneus]

  ## Build the population 

  trace_ker = H.Trace(tauker,nneus,H.ForDynamics())

  popstate = H.PopulationStateMixedExp(trains,trace_ker)
  connection = H.ConnectionExpKernel(noweights,trace_ker)
  population = H.PopulationMixedExp(popstate,connection,noinputs)

  n_spikes = round(Integer,rates_start*Tend*0.9*nneus)
  recorder = H.RecFullTrain(n_spikes,1)
  network = H.RecurrentNetworkExpKernel(population,recorder)

  function simulate!(network,num_spikes)
    t_now = 0.0
    H.reset!(network) # clear spike trains etc
    for _ in 1:num_spikes
      t_now = H.dynamics_step!(t_now,network)
    end
    return t_now
  end

  Tmax = simulate!(network,n_spikes)

  trains_out = H.get_trains(recorder,nneus)

  # no interactions, trains are identical to forced spiking
  @test all( trains[1][1:length(trains_out[1])] .== trains_out[1])
  @test all( trains[2][1:length(trains_out[2])] .== trains_out[2])
  @test all( trains[3][1:length(trains_out[3])] .== trains_out[3])

  # now include interactions
  someweights = [  0 0 0 ; 0.5 0 0 ; 1.0 0 0.0]
  copy!(connection.weights,someweights)

  n_spikes = round(Integer,(1+1.5+2)rates_start*Tend*0.9)
  recorder = H.RecFullTrain(n_spikes,1)
  network = H.RecurrentNetworkExpKernel(population,recorder)

  Tmax = simulate!(network,n_spikes)
  trains_out2 = H.get_trains(recorder,nneus)
  therates = H.numerical_rates(recorder,nneus,Tmax)

  # unconnected neuron, only forced spikes
  @test all( trains[1][1:length(trains_out2[1])] .== trains_out2[1])
  # neuron receiving 0.5 input spikes 1.5 times as much 
  @test isapprox(therates[2],1.5*rates_start;rtol=0.2)
  # neuron receiving 1.0 input spikes twice as much
  @test isapprox(therates[3],2*rates_start;rtol=0.2)
end



@testset "Exp kernel Vs 2D model" begin
  ne = 5
  ni = 5
  N = ne+ni
  idxe = 1:ne
  idxi = idxe[end] .+ (1:ni)
  wee = 1.4
  wie = 1.0
  wei = -1.0
  wii = -0.1
  wmat = fill(0.0,(N,N))

  wmat[idxe,idxe] .=   wee/(ne-1)
  wmat[idxi,idxe] .=   wie/ne
  wmat[idxe,idxi] .=   wei/ni
  wmat[idxi,idxi] .=   wii/(ni-1)
  wmat[diagind(wmat)] .= 0.0

  wmat_ee = wmat[idxe,idxe]
  wmat_ie = wmat[idxi,idxe]
  wmat_ei = abs.(wmat[idxe,idxi])
  wmat_ii = abs.(wmat[idxi,idxi])

  ??e = 0.5
  ??i = 0.5

  pse,trae = H.population_state_exp_and_trace(ne,??e)
  psi,trai = H.population_state_exp_and_trace_inhibitory(ni,??i)

  conn_ee = H.ConnectionExpKernel(wmat_ee,trae)
  conn_ie = H.ConnectionExpKernel(wmat_ie,trae)
  conn_ei = H.ConnectionExpKernel(wmat_ei,trai)
  conn_ii = H.ConnectionExpKernel(wmat_ii,trai)

  in_e = 60.0
  in_i = 50.0

  r0e = fill(in_e,ne)
  r0i = fill(in_i,ni)
  r0full = vcat(r0e,r0i)

  rates_an = rates_analytic(wmat,r0full)

  population_e = H.PopulationExpKernel(pse,r0e,(conn_ei,psi),(conn_ee,pse);
    nonlinearity=H.NLRmax(300.))
  population_i = H.PopulationExpKernel(psi,r0i,(conn_ii,psi),(conn_ie,pse);
    nonlinearity=H.NLRmax(300.))

  n_spikes = 10_000
  recorder = H.RecFullTrain(n_spikes+1,2)
  network = H.RecurrentNetworkExpKernel((population_e,population_i),(recorder,))


  function simulate!(network,num_spikes;initial_e=nothing,initial_i=nothing)
    t_now = 0.0
    H.reset!(network) # clear spike trains etc
    H.reset!.((pse,psi))
    H.set_initial_rates!(population_e,initial_e)
    H.set_initial_rates!(population_i,initial_i)
    for _ in 1:num_spikes
      t_now = H.dynamics_step!(t_now,network)
    end
    return t_now
  end

    t_end =simulate!(network,n_spikes;initial_e=fill(30.,ne),initial_i=fill(40.,ni))

  rates_num_e = H.numerical_rates(recorder,ne,t_end;pop_idx=1)
  rates_num_i = H.numerical_rates(recorder,ni,t_end;pop_idx=2)
  rate_num_e = mean(rates_num_e[1:ne])
  rate_num_i = mean(rates_num_i[1:ne])
  @test all(isapprox.(rate_num_e,rates_num_e;rtol=0.2))
  @test isapprox(rate_num_e,rates_an[1];rtol=0.2)
  @test all(isapprox.(rate_num_i,rates_num_i;rtol=0.2))
  @test isapprox(rate_num_i,rates_an[end];rtol=0.2)
end

##