var documenterSearchIndex = {"docs":
[{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"EditURL = \"https://github.com/dylanfesta/HawkesSimulator.jl/blob/master/examples/alphadelay.jl\"","category":"page"},{"location":"alphadelay/#D-Hawkes-process-with-delayed-alpha-kernel","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"","category":"section"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"In this example, I simulate either a 1D or a 2D Hawkes process, with a delayed-alpha interaction kernel","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"g(t) = H(t-t_textdelay)   frac(t-t_textdelay)tau^2 \n expleft(- frac(t-t_textdelay)tau right)","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"where H(x) is the Heaviside function: H(x)=0 for x0, H(x)=1 for tgeq 0.","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"Kernels are always normalized so that their integral is 1.","category":"page"},{"location":"alphadelay/#Initialization","page":"1D Hawkes process with delayed alpha kernel","title":"Initialization","text":"","category":"section"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"using LinearAlgebra,Statistics,StatsBase,Distributions\nusing Plots,NamedColors ; theme(:default) #; plotlyjs()\nusing FFTW\n\nusing ProgressMeter\nusing Random\nRandom.seed!(0)\n\nusing HawkesSimulator; const global H = HawkesSimulator\n\n#","category":"page"},{"location":"alphadelay/#Define-and-visualize-the-kernel","page":"1D Hawkes process with delayed alpha kernel","title":"Define and visualize the kernel","text":"","category":"section"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"mytau = 0.5\nmydelay = 2.0\nmyw = fill(0.85,(1,1)) # 1x1 matrix\nmyinput = [0.5,]       # 1-dim vector\n\nker = H.KernelAlphaDelay(mytau,mydelay)\n\nplt = let ts = range(-0.5,8;length=150)\n  y = [H.interaction_kernel(_t,ker) for _t in ts]\n  plt=plot(ts,y ; linewidth=3,leg=false,xlabel=\"time (s)\",\n      ylabel=\"interaction kernel\")\n  ymax=ylims()[2]\n  plot!(plt,[0,0],[0,ymax];linestyle=:dash,linecolor=:black)\nend;\nplot(plt)","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"Note that the kernel starts after zero, according to the delay indicated.","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"As a side note: in order to simulate Hawkes processes, one always needs to define a non-increasing upper limit to the kernel. This is what it looks like for this kernel.","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"plt = let  ts = range(-0.5,8;length=150)\n  y = [H.interaction_kernel(_t,ker) for _t in ts]\n  yu = [H.interaction_kernel_upper(_t,ker) for _t in ts]\n  plt = plot(ts,y ; linewidth=3,xlabel=\"time (s)\",\n      ylabel=\"interaction kernel\", label=\"true kernel\")\n  plot!(plt, ts,yu ; linewidth=2, label=\"upper limit\", linestyle=:dash)\n  ymax=ylims()[2]\n  plot!(plt,[0,0],[0,ymax];linestyle=:dash,linecolor=:black,label=\"\")\nend;\nplot(plt)","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"the closer the upper limit is to the true kernel, the more efficient the simulation.","category":"page"},{"location":"alphadelay/#Build-the-network-and-run-it","page":"1D Hawkes process with delayed alpha kernel","title":"Build the network and run it","text":"","category":"section"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"I compare the final rate with what I expect from the analytic solution (see first example file)","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"pops = H.PopulationState(ker,1)\nntw = H.RecurrentNetwork(pops,myw,myinput)\n\nfunction run_simulation!(network,num_spikes,\n    t_flush_trigger=300.0,t_flush=100.0)\n  t_now = 0.0\n  H.reset!(network) # clear spike trains etc\n  @showprogress \"Running Hawkes process...\" for _ in 1:num_spikes\n    t_now = H.dynamics_step_singlepopulation!(t_now,network)\n    H.flush_trains!(network,t_flush_trigger;Tflush=t_flush)\n  end\n  H.flush_trains!(network) # flush everything into history\n  return t_now\nend\n\nn_spikes = 80_000\nTmax = run_simulation!(ntw,n_spikes);\n\nratenum = H.numerical_rates(pops)[1]\nrate_analytic = (I-myw)\\myinput\nrate_analytic = rate_analytic[1] # from 1-dim vector to scalar\n\n@info \"Mean rate -  numerical $(round(ratenum;digits=2)), analytic  $(round(rate_analytic;digits=2))\"\n\n#","category":"page"},{"location":"alphadelay/#Covariance-density","page":"1D Hawkes process with delayed alpha kernel","title":"Covariance density","text":"","category":"section"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"First, compute covariance density numerically for a reasonable time step","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"mytrain = pops.trains_history[1]\nmydt = 0.1\nmyτmax = 60.0\nmytaus = H.get_times(mydt,myτmax)\nntaus = length(mytaus)\ncov_num = H.covariance_self_numerical(mytrain,mydt,myτmax);\nnothing #hide","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"now compute covariance density  analytically (as in Hawkes models), at higher resolution, and compare analytic and numeric","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"Note that the high resolution is not just for a better plot, but also to ensure the result is more precise when we move from frequency domain (Fourier transforms) to time domain.","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"function four_high_res(dt::Real,Tmax::Real) # higher time resolution, longer time\n  k1,k2 = 2 , 0.01\n  myτmax = Tmax * k1\n  dt *= k2\n  mytaus = H.get_times(dt,myτmax)\n  nkeep = div(length(mytaus),k1)\n  myfreq = H.get_frequencies_centerzero(dt,myτmax)\n  gfou = myw[1,1] .* H.interaction_kernel_fourier.(myfreq,Ref(ker)) |> ifftshift\n  ffou = let r=rate_analytic\n    covf(g) = r/((1-g)*(1-g'))\n    map(covf,gfou)\n  end\n  retf = real.(ifft(ffou))\n  retf[2:end] ./= dt # first element is rate\n  return mytaus[1:nkeep],retf[1:nkeep]\nend\n\ntaush,covfou=four_high_res(mydt,myτmax)\n\nplt= let plt = plot(xlabel=\"time delay (s)\",ylabel=\"Covariance density\")\n  plot!(plt,mytaus[2:end], cov_num[2:end] ; linewidth=3, label=\"numerical\" )\n  plot!(plt,taush[2:end],covfou[2:end]; label=\"analytic\",linewidth=3,linestyle=:dash)\nend;\n\nplot(plt)","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"Analytics and numerics match quite well.","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"THE END","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"","category":"page"},{"location":"alphadelay/","page":"1D Hawkes process with delayed alpha kernel","title":"1D Hawkes process with delayed alpha kernel","text":"This page was generated using Literate.jl.","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"EditURL = \"https://github.com/dylanfesta/HawkesSimulator.jl/blob/master/examples/exp_1and2D.jl\"","category":"page"},{"location":"exp_1and2D/#D-and-2D-Hawkes-processes-with-exponential-kernel","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"","category":"section"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"In this example, I simulate first a 1D self-exciting Hawkes process and then a 2D one. The interaction kernel is an exponentially decaying function, defined as:","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"g(t) = H(t)   frac1tau  expleft(- fracttau right)","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"where H(t) is the Heaviside function: zero for t0, one for tgeq 0.","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"Note that Kernels are always normalized so that their integral between -infty and +infty is 1.","category":"page"},{"location":"exp_1and2D/#Initialization","page":"1D and 2D Hawkes processes with exponential kernel","title":"Initialization","text":"","category":"section"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"using LinearAlgebra,Statistics,StatsBase,Distributions\nusing Plots,NamedColors ; theme(:default) ; gr()\nusing FFTW\n\nusing ProgressMeter\nusing Random\nRandom.seed!(0)\n\nusing HawkesSimulator; const global H = HawkesSimulator\n\n\"\"\"\n    onedmat(x::R) where R\nGenerates a 1-by-1 Matrix{R} that contains `x` as only element\n\"\"\"\nfunction onedmat(x::R) where R\n  return cat(x;dims=2)\nend;\nnothing #hide","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"First I define the kernel, and the self-interaction weight. The kernel is defined through a \"Population\": all neurons in the same population have the same kernel.","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"myw is a scaling factor (the weight of the autaptic connection). The baseline rate is given by myinput","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"mytau = 0.5  # kernel time constant\nmyw = onedmat(0.85) # weight: this needs to be a matrix\nmyinput = [0.7,] # this needs to be a vector\nmykernel = H.KernelExp(mytau);\nnothing #hide","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"This is the plot of the (self) interaction kernel (before  the scaling by myw)","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"theplot = let ts = range(-1.0,5;length=150),\n  y = map(t->H.interaction_kernel(t,mykernel) , ts )\n  plot(ts , y ; linewidth=3,leg=false,xlabel=\"time (s)\",\n     ylabel=\"interaction kernel\")\nend;\nplot(theplot)","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"Now I build the network, using the simplified constructor","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"popstate = H.PopulationState(mykernel,1)\nntw = H.RecurrentNetwork(popstate,myw,myinput);\nnothing #hide","category":"page"},{"location":"exp_1and2D/#Simulation","page":"1D and 2D Hawkes processes with exponential kernel","title":"Simulation","text":"","category":"section"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"The length of the simulation is measured by the total number of spikes here called num_spikes. The function flush_trains!(...) is used to store older spikes as history and let them be ignored by the kernels. The time parameters should be regulated based on the kernel shape.","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"function run_simulation!(network,num_spikes,\n    t_flush_trigger=300.0,t_flush=100.0)\n  t_now = 0.0\n  H.reset!(network) # clear spike trains etc\n  @showprogress \"Running Hawkes process...\" for _ in 1:num_spikes\n    t_now = H.dynamics_step_singlepopulation!(t_now,network)\n    H.flush_trains!(network,t_flush_trigger;Tflush=t_flush)\n  end\n  H.flush_trains!(network) # flush everything into history\n  return t_now\nend\n\nn_spikes = 100_000\nTmax = run_simulation!(ntw,n_spikes)\nratenum = H.numerical_rates(popstate;Tend=Tmax)[1]\n@info \"Simulation completed, mean rate $(round(ratenum;digits=2)) Hz\"\n\n#","category":"page"},{"location":"exp_1and2D/#Visualize-raster-plot-of-the-events","page":"1D and 2D Hawkes processes with exponential kernel","title":"Visualize raster plot of the events","text":"","category":"section"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"the raster plot shows some correlation between the neural activities neuron one (lower row) excites neuron two (upper row)","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"function rasterplot(tlims = (1100.,1120.) )\n  _train = popstate.trains_history[1]\n  plt=plot()\n  train = filter(t-> tlims[1]< t < tlims[2],_train)\n  nspk = length(train)\n  scatter!(plt,train,fill(0.1,nspk),markersize=30,\n      markercolor=:black,markershape=:vline,leg=false)\n  return plot!(plt,ylims=(0.0,0.2),xlabel=\"time (s)\")\nend\n\nrasterplot()","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"event times are always stored in pops.trains_history","category":"page"},{"location":"exp_1and2D/#Plot-the-instantaneous-rate","page":"1D and 2D Hawkes processes with exponential kernel","title":"Plot the instantaneous rate","text":"","category":"section"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"This is the probability of a spike given the past activity up until that moment. It is usually denoted by lambda^*(t).","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"function get_insta_rate(t,popstate)\n  _train = popstate.trains_history[1]\n  myinput[1] + H.interaction(t,_train,myw[1],popstate.unittype)\nend\nfunction plot_instarate(popstate,tlims=(1100,1120))\n  tplot = range(tlims...;length=100)\n  _train = popstate.trains_history[1]\n  tspk = filter(t-> tlims[1]<=t<=tlims[2],_train) # add the exact spiketimes for cleaner plot\n  tplot = sort(vcat(tplot,tspk,tspk .- 1E-4))\n  plt=plot(xlabel=\"time (s)\",ylabel=\"instantaneous rate\")\n  plot!(plt,t->get_insta_rate(t,popstate),tplot;linewidth=2,color=:black)\n  scatter!(t->get_insta_rate(t,popstate),tspk;leg=false)\n  avg_rate = H.numerical_rates(popstate;Tend=Tmax)[1]\n  ylim2 = ylims()[2]\n  plot!(plt,tplot, fill(avg_rate,length(tplot)),\n     color=:red,linestyle=:dash,ylims=(0,ylim2))\nend\nplot_instarate(popstate)","category":"page"},{"location":"exp_1and2D/#Plot-the-total-event-counts","page":"1D and 2D Hawkes processes with exponential kernel","title":"Plot the total event counts","text":"","category":"section"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"Total number of events as a function of time. It grows linearly, and the steepness is pretty much the rate.","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"function plot_counts(tlims=(0.,1000.))\n  avg_rate = H.numerical_rates(popstate;Tend=Tmax)[1]\n  tplot = range(tlims...;length=100)\n  _train = popstate.trains_history[1]\n  nevents(tnow::Real) = count(t-> t <= tnow,_train)\n  plt=plot(xlabel=\"time (s)\",ylabel=\"number of events\",leg=false)\n  plot!(plt,tplot,nevents.(tplot),color=:black,linewidth=2)\n  plot!(plt,tplot , tplot .* avg_rate,color=:red,linestyle=:dash)\nend\nplot_counts()","category":"page"},{"location":"exp_1and2D/#Rate","page":"1D and 2D Hawkes processes with exponential kernel","title":"Rate","text":"","category":"section"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"Now I compare the numerical rate with the analytic solution.","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"The analytic rate corresponds to the stationary solution of a linear dynamical system (assumung all stationary rates are above zero).","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"fracmathrm d mathbf rmathrm d t = - mathbf r + Wr + mathbf h quad\nqquad  r_infty = (I-W)^-1  mathbf h","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"rate_analytic = inv(I-myw)*myinput\nrate_analytic = rate_analytic[1] # 1-D , just a scalar\n\n@info \"Mean rate -  numerical $(round(ratenum;digits=2)), analytic  $(round(rate_analytic;digits=2))\"\n\n#","category":"page"},{"location":"exp_1and2D/#Covariance-density","page":"1D and 2D Hawkes processes with exponential kernel","title":"Covariance density","text":"","category":"section"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"TODO : write definition","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"I compute the covariance density it numerically. The time inteval mydt should not be too small.","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"mytrain = popstate.trains_history[1]\nmydt = 0.1\nmyτmax = 25.0\nmytaus = H.get_times(mydt,myτmax)\nntaus = length(mytaus)\ncov_num = H.covariance_self_numerical(mytrain,mydt,myτmax);\n\n#","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"Now I compute the covariance density analytically, at higher resolution, and I compare the two.","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"function four_high_res(dt::Real,Tmax::Real) # higher time resolution, longer time\n  k1,k2 = 2 , 0.01\n  myτmax = Tmax * k1\n  dt *= k2\n  mytaus = H.get_times(dt,myτmax)\n  nkeep = div(length(mytaus),k1)\n  myfreq = H.get_frequencies_centerzero(dt,myτmax)\n  gfou = myw[1,1] .* H.interaction_kernel_fourier.(myfreq,Ref(popstate)) |> ifftshift\n  ffou = let r=rate_analytic\n    covf(g) = r/((1-g)*(1-g'))\n    map(covf,gfou)\n  end\n  retf = real.(ifft(ffou)) ./ dt\n  retf[1] *= dt  # first element is rate\n  return mytaus[1:nkeep],retf[1:nkeep]\nend\n\n(taush,covfou)=four_high_res(mydt,myτmax)\n\nfunction doplot()\n  plt = plot(xlabel=\"time delay (s)\",ylabel=\"Covariance density\")\n  plot!(plt,mytaus[2:end], cov_num[2:end] ; linewidth=3, label=\"simulation\" )\n  plot!(plt,taush[2:end],covfou[2:end]; label=\"analytic\",linewidth=3,linestyle=:dash)\n  return plt\nend\ndoplot()","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"1D system completed !","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"#","category":"page"},{"location":"exp_1and2D/#Same-results,-but-in-a-2D-system","page":"1D and 2D Hawkes processes with exponential kernel","title":"Same results, but in a 2D system","text":"","category":"section"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"myτ = 1/2.33\nmywmat = [ 0.31   0.3\n           0.9  0.15 ]\nmyin = [1.0,0.1]\np1 = H.KernelExp(myτ)\nps1 = H.PopulationState(p1,2)\nntw = H.RecurrentNetwork(ps1,mywmat,myin)","category":"page"},{"location":"exp_1and2D/#Start-the-simulation","page":"1D and 2D Hawkes processes with exponential kernel","title":"Start the simulation","text":"","category":"section"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"the function run_simulation!(...) has been defined above Note that n_spikes is the total number of spikes among all units in the system.","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"n_spikes = 500_000\n\nTmax = run_simulation!(ntw,n_spikes,100.0,10.0);\nnothing #hide","category":"page"},{"location":"exp_1and2D/#Check-the-rates","page":"1D and 2D Hawkes processes with exponential kernel","title":"Check the rates","text":"","category":"section"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"The analytic rate is from  Eq between 6 and  7 in Hawkes 1971","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"num_rates = H.numerical_rates(ps1)\nmyspikes_both = ps1.trains_history\n\nratefou = let G0 =  mywmat .* H.interaction_kernel_fourier(0,p1)\n  inv(I-G0)*myin |> real\nend\nrate_analytic = inv(I-mywmat)*myin\n\n@info \"Total duration $(round(Tmax;digits=1)) s\"\n@info \"Rates are $(round.(num_rates;digits=2))\"\n@info \"Analytic rates are $(round.(rate_analytic;digits=2)) Hz\"","category":"page"},{"location":"exp_1and2D/#Covariance-density-2","page":"1D and 2D Hawkes processes with exponential kernel","title":"Covariance density","text":"","category":"section"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"there are 4 combinations, therefore I will compare 4 lines.","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"mydt = 0.1\nmyτmax = 15.0\nmytaus = H.get_times(mydt,myτmax)\nntaus = length(mytaus)\ncov_num = H.covariance_density_numerical(myspikes_both,mydt,myτmax)\n\nfunction doplot()\n  plt=plot(xlabel=\"time delay (s)\",ylabel=\"Covariance density\")\n  for i in 1:2, j in 1:2\n    plot!(plt,mytaus[2:end-1],cov_num[i,j,2:end-1], linewidth = 3, label=\"cov $i-$j\")\n  end\n  return plt\nend\n\ndoplot()","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"The analytic solution is eq 12 from Hawkes 1971","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"function four_high_res(dt::Real,Tmax::Real)\n  k1 = 2\n  k2 = 0.005\n  myτmax,mydt = Tmax * k1, dt*k2\n  mytaus = H.get_times(mydt,myτmax)\n  nkeep = div(length(mytaus),k1)\n  myfreq = H.get_frequencies_centerzero(mydt,myτmax)\n  G_omega = map(mywmat) do w\n    ifftshift( w .* H.interaction_kernel_fourier.(myfreq,Ref(p1)))\n  end\n  D = Diagonal(ratefou)\n  M = Array{ComplexF64}(undef,2,2,length(myfreq))\n  Mt = similar(M,Float64)\n  for i in eachindex(myfreq)\n    G = getindex.(G_omega,i)\n    M[:,:,i] = (I-G)\\D/(I-G')\n  end\n  for i in 1:2,j in 1:2\n    Mt[i,j,:] = real.(ifft(M[i,j,:]))\n    Mt[i,j,2:end] ./= mydt # diagonal of t=0 contains the rate\n  end\n  return mytaus[1:nkeep],Mt[:,:,1:nkeep]\nend\n\ntaush,Cfou=four_high_res(mydt,myτmax)\n\nfunction oneplot(i,j)\n  plt=plot(xlabel=\"time delay (s)\",ylabel=\"Covariance density\",title=\"cov $i - $j\")\n  plot!(plt,mytaus[2:end],cov_num[i,j,2:end] ; linewidth = 3, label=\"simulation\")\n  plot!(plt,taush[2:end],Cfou[i,j,2:end]; linestyle=:dash, linewidth=3, label=\"analytic\")\nend","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"1","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"oneplot(1,1)","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"2","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"oneplot(1,2)","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"3","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"oneplot(2,1)","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"4","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"oneplot(2,2)","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"THE END","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"","category":"page"},{"location":"exp_1and2D/","page":"1D and 2D Hawkes processes with exponential kernel","title":"1D and 2D Hawkes processes with exponential kernel","text":"This page was generated using Literate.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = HawkesSimulator","category":"page"},{"location":"#Hawkes-Processes-Simulator","page":"Home","title":"Hawkes Processes Simulator","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"","page":"Home","title":"Home","text":"warning: Warning\nThe documentation is still missing. Please see the \"examples\" section for usage.","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"1D and 2D, exponential kernel\n1D delayed-alpha kernel","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [HawkesSimulator]","category":"page"},{"location":"#HawkesSimulator.draw_spike_raster-Union{Tuple{C}, Tuple{Vector{Vector{Float64}}, Real, Real}} where C<:ColorTypes.Color","page":"Home","title":"HawkesSimulator.draw_spike_raster","text":" draw_spike_raster(trains::Vector{Vector{Float64}},\n  dt::Real,Tend::Real;\n  Tstart::Real=0.0,\n  spike_size::Integer = 5,\n  spike_separator::Integer = 1,\n  background_color::Color=RGB(1.,1.,1.),\n  spike_colors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),\n  max_size::Real=1E4) where C<:Color\n\nDraws a matrix that contains the raster plot of the spike train.\n\nArguments\n\nTrains :  Vector of spike trains. The order of the vector corresponds to \n\nthe order of the plot. First element is at the top, second is second row, etc.\n\ndt : time interval representing one horizontal pixel  \nTend : final time to be considered\n\nOptional arguments\n\nTstart : starting time\nmax_size : throws an error if image is larger than this number (in pixels)\nspike_size : heigh of spike (in pixels)\nspike_separator : space between spikes, and vertical padding\nbackground_color : self-explanatory\nspike_colors : if a single color, color of all spikes, if vector of colors, \n\ncolor for each neuron (length should be same as number of neurons)\n\nreturns\n\nraster_matrix::Matrix{Color} you can save it as a png file\n\n\n\n\n\n","category":"method"},{"location":"#HawkesSimulator.dynamics_step_singlepopulation!-Tuple{Real, HawkesSimulator.RecurrentNetwork}","page":"Home","title":"HawkesSimulator.dynamics_step_singlepopulation!","text":"dynamics_step_singlepopulation!(t_now::Real,ntw::RecurrentNetwork)\n\nIterates a one-population network up until its next spike time. This is done by computing a next spike proposal for each neuron, and then picking the one that happens sooner. This spike is then added to the  spiketrain for that neuron. The fundtion returns the new current time of the simulation.\n\nFor long simulations, this functions should be called jointly with  flush_trains!. Otherwise the spike trains will keep growing, making the  propagation of signals extremely cumbersome.\n\nArguments\n\nt_now - Current time of the simulation ntw   - The network\n\nReturns\n\nt_now_new - the new current time of the simulation\n\n\n\n\n\n","category":"method"},{"location":"#HawkesSimulator.flush_trains!-Tuple{HawkesSimulator.PopulationState, Real}","page":"Home","title":"HawkesSimulator.flush_trains!","text":"flush_trains!(ps::PopulationState,Ttrigger::Real;\n    Tflush::Union{Real,Nothing}=nothing)\n\nSpike history is spiketimes that do not interact with the kernel (because too old)         This function compares most recent spike with spike history, if enough time has passed   (measured by Ttrigger) it flushes the spiketrain up to Tflush into the history.\n\n\n\n\n\n","category":"method"},{"location":"#HawkesSimulator.warmup_step!-Tuple{Real, HawkesSimulator.RecurrentNetwork, Union{Vector{Vector{Float64}}, Vector{Float64}}}","page":"Home","title":"HawkesSimulator.warmup_step!","text":"warmupstep!(tnow::Real,ntw::RecurrentNetwork,     warmuprates::Union{Vector{Float64},Vector{Vector{Float64}}}) -> tend\n\nIn the warmup phase, all neurons fire as independent Poisson process  with a rate set by warmup_rates.   This is useful to quick-start the network, or set itinitial conditions that are far from the stable point.\n\nArguments\n\nt_now - current time\nntw  - the network (warning: weights and kernels are entirely ignored here)\nwarmup_rates - the desired stationary rates. In a one-population network,   it is a vector with the desired rates. In a multi-population network,   is a collection of vectors, where each vector refers to one population.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"TODO : add other  file. [comment]: <> ( 1. 2D delayed-alpha interactions, and non-delayed autapses )","category":"page"}]
}
