using Distributed
using ClusterManagers
addprocs(SlurmManager(100))
@everywhere begin
using Pkg
Pkg.instantiate()
using FFTW
using Random
using Statistics
using LsqFit
using CSV
using DelimitedFiles

function distance(p,q)::Float64
	return sqrt(p^2+q^2)
end

function initial_lattice(L::Int64;cold_start=true)::Array{Int64,2}
	lattice=ones(Int64,L,L)
	if cold_start==false
		lattice=rand([1,-1],L,L)
	end
	return lattice
end

function metropolis_step(L::Int64,T::Float64,lattice::Array{Int64,2})::Array{Int64,2}
	del_E=0.0
	start_i,start_j=rand(1:L,1,2)
	flip_spin=lattice[start_i,start_j]
	for (i,j) in [(1,0),(0,1),(-1,0),(0,-1)]
		del_E_j+= 2.0*flip_spin*lattice[mod1(start_i+i,L),mod1(start_j+j,L)]
	end
	if del_E<=0.0
		lattice[start_i,start_j]*=-1
	elseif rand()<exp(-del_E/T)
		lattice[start_i,start_j]*=-1
	end
	return lattice
end

function correlations(L::Int64,lattice::Array{Int64,2})::Array{Float64,2}
	if L%2==0
		even_dim=convert(Int64,(((L/2)^2-1)+ ((L/2)+1) + (L/2) ))
		correlations_list=zeros(Float64,even_dim,2)
		count=0
		for p=0:convert(Int64,((L/2)-1)),q=0:convert(Int64,((L/2)-1))
			dist=0
			correlation=0
			dist=distance(p,q)
			if dist>=1.0
				count+=1
				for i=1:L,j=1:L
					correlation+=lattice[i,j]*lattice[mod1(i+p,L),mod1(j+q,L)]
				end
				correlation/=float(L*L)
				correlations_list[count,1]=dist
				correlations_list[count,2]=correlation
			end
		end
		for p=convert(Int64,L/2),q=0:convert(Int64,L/2)
			dist=0
			correlation=0
			dist=distance(p,q)
			if dist>=1.0
				count+=1
				for i=1:Int(L/2),j=1:Int(L/2)
					correlation+=lattice[i,j]*lattice[mod1(i+p,L),mod1(j+q,L)]
				end
				correlation/=float(L*L)
				correlations_list[count,1]=dist
				correlations_list[count,2]=correlation
			end
		end
		for p=0:convert(Int64,L/2-1),q=convert(Int64,L/2)
			dist=0
			correlation=0
			dist=distance(p,q)
			if dist>=1.0
				count+=1
				for i=1:Int(L/2),j=1:Int(L/2)
					correlation+=lattice[i,j]*lattice[mod1(i+p,L),mod1(j+q,L)]
				end
				correlation/=float(L*L)
				correlations_list[count,1]=dist
				correlations_list[count,2]=correlation
			end
		end

	else
		odd_dim=convert(Int64,(((L-1)/2)+1)^2-1)
		correlations_list=zeros(Float64,odd_dim,2)
		count=0
		for p=0:convert(Int64,((L-1)/2)),q=0:convert(Int64,((L-1)/2))
			dist=0
			correlation=0
			dist=distance(p,q)
			if dist>=1.0
				count+=1
				for i=1:L,j=1:L
					correlation=lattice[i,j]*lattice[mod1(i+p,L),mod1(j+q,L)]
				end
				correlation/=float(L*L)
				correlations_list[count,1]=dist
				correlations_list[count,2]=correlation
			end
		end
	end
	return correlations_list
end

function calc_energy(L::Int64,lattice::Array{Int64,2})::Float64
	energy=0.0
	for i=1:L,j=1:L
		C1 +=lattice[i,j]*(lattice[mod1(i+1,L),j]+lattice[i,mod1(j+1,L)])
	end
	C1 /=float(L*L*L)
	energy = -1.0*C1
	return energy
end

function autocorrelation(observable::Array{Float64,1})::Int64
	fm=FFTW.fft(observable[:].-mean(observable[:]))./sqrt(length(observable[:])) #fourier transform of mag
	fm2=(abs.(fm)).^2 #autocorrelation of the fourieer transformed mag
	cm=real(FFTW.ifft(fm2)) #inverse FT of the fm2, i.e. autocorrelation funciton of Observable(t)
	cm_2= cm ./ cm[1] #autocorrelation normalized
	log_cm_2 =[]#log list of autocorelation function(time-step)
	j=1
	while cm_2[j] > 0
		push!(log_cm_2,log(cm_2[j]))
		j = j+1
	end
	t = collect(LinRange(0,length(log_cm_2)-1,length(log_cm_2)))
	@. line(t,p)=p[1]*t+p[2]
	p0=[-0.5,-0.5]
	fit = curve_fit(line,t,log_cm_2,p0)
	p=fit.param
	tau = -1 / p[1]
	correlation=ceil(Int64,tau)#making it in multiples of 10 because saving every 10th sweep
	return correlation
end

function dnn_simulation(L::Int64,T::Float64,sweeps::Int64)
	one_sweep=L*L
	mkpath("/scratch/fermi/jawla/2d_ising_dnn_data_L=$(L)_T=$(T)")
	lattice=initial_lattice(L,cold_start=false)
	open("/scratch/fermi/jawla/Ongoing/2d_ising_dnn_data_L=$(L)_T=$(T)/energy_timeseries.txt","w") do f
		for i in 1:sweeps
			for step in 1:one_sweep
				lattice_1=metropolis_step(L,T,lattice)
				lattice=lattice_1
			end
			energy=calc_energy(L,lattice)
			write(f,"$(energy)\n")
			if i>= 20000 && i%1000==0
				corr_list=correlations(L,lattice)
				writedlm("/scratch/fermi/jawla/Ongoing/2d_ising_dnn_data_L=$(L)_T=$(T)/sweep_data/sweep=$(i).txt", corr_list, ',')
			end
		end
	end
	continue_dnn_or_not(L,T,sweeps,lattice)
	return nothing
end

function continue_dnn_or_not(L::Int64,T::Float64,sweeps::Int64,lattice::Array{Int64,2})
	E_path="/scratch/fermi/jawla/Ongoing/2d_ising_dnn_data_L=$(L)_T=$(T)/energy_timeseries.txt"
	array=Matrix(CSV.read(E_path,header=false))
	array=reshape(array,length(array))
	sweeps=length(array)
	tau=autocorrelation(array)
	open("/scratch/fermi/jawla/Ongoing/2d_ising_dnn_data_L=$(L)_T=$(T)/tau.txt","w") do f
		write(f,"$(tau)")
	end
	if 25*tau<=sweeps
		mkpath("/scratch/fermi/jawla/Data/2d_ising_dnn_data_L=$(L)_T=$(T)")
		mv("/scratch/fermi/jawla/Ongoing/2d_ising_dnn_data_L=$(L)_T=$(T)","/scratch/fermi/jawla/Data/2d_ising_dnn_data_L=$(L)_T=$(T)",force=true)
	else
		new_sweeps=30*tau
		continue_dnn_simulation(L,T,new_sweeps,sweeps,lattice)
	end
end

function continue_dnn_simulation(L::Int64,T::Float64,extra_sweeps::Int64,sweeps::Int64,lattice::Array{Int64,2})
	one_sweep=L*L
	open("/scratch/fermi/jawla/Ongoing/2d_ising_dnn_data_L=$(L)_T=$(T)/energy_timeseries.txt","a") do f
		for i in sweeps+1:extra_sweeps
			for step in 1:one_sweep
				lattice_1=metropolis_step(L,T,lattice)
				lattice=lattice_1
			end
			energy=calc_energy(L,lattice)
			write(f,"$(energy)\n")
			if i>= 20000 && i%1000==0
				corr_list=correlations(L,lattice)
				writedlm("/scratch/fermi/jawla/Ongoing/2d_ising_dnn_data_L=$(L)_T=$(T)/sweep_data/sweep=$(i).txt", corr_list, ',')
			end
		end
	end
	sweeps=extra_sweeps
	continue_dnn_or_not(L,T,sweeps,lattice)
	return nothing
end
end
lattices=[8]
# temp1=collect(range(0.0,stop=1.65, step=0.033))
# temp2=collect(range(3.0,stop=4.65, step=0.033))
# temp=vcat(temp1,temp2)
temp=[0,Inf]
comb=[[a,b] for a in lattices,b in temp]
z=reshape(comb,length(comb))
pmap(dnn_simulation,[Int(i[1]) for i in z],[i[2] for i in z],repeat([10000],length(comb)))
