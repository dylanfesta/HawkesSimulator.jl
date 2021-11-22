using HawkesSimulator ; global const H = HawkesSimulator
using Test
using Distributions,Statistics,StatsBase,LinearAlgebra
using QuadGK
using Random ; Random.seed!(0)


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