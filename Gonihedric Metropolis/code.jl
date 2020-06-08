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

function distance(p,q,r)::Float64
	return sqrt(p^2+q^2+r^2)
end

function initial_lattice(L::Int64;cold_start=true)::Array{Int64,3}
	lattice=ones(Int64,L,L,L)
	if cold_start==false
		lattice=rand([1,-1],L,L,L)
	end
	return lattice
end

function metropolis_step(L::Int64,kappa::Float64,T::Float64,lattice::Array{Int64,3})::Array{Int64,3}
	del_E_j,del_E_k,del_E_l,del_E=0.0,0.0,0.0,0.0
	constant_j,constant_k,constant_l=2*kappa,-kappa/2.0,(1-kappa)/2.0
	start_i,start_j,start_k=rand(1:L,1,3)
	flip_spin=lattice[start_i,start_j,start_k]
	for (i,j,k) in [(1,0,0),(0,1,0),(0,0,1),(-1,0,0),(0,-1,0),(0,0,-1)]
		del_E_j+= flip_spin*lattice[mod1(start_i+i,L),mod1(start_j+j,L),mod1(start_k+k,L)]
	end
	for (i,j,k) in [(1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),(-1,1,0),(1,1,0),(1,-1,0),(-1,-1,0),(0,-1,1),(0,-1,-1),(0,1,-1),(0,1,1)]
		del_E_k+= flip_spin*lattice[mod1(start_i+i,L),mod1(start_j+j,L),mod1(start_k+k,L)]
	end
	for (i,j,k,l,m,n,p,q,r) in [(0,0,-1,1,0,-1,1,0,0),(1,0,0,0,-1,0,1,-1,0),(0,0,-1,0,-1,-1,0,-1,0),(0,0,1,1,0,1,1,0,0),(1,0,0,1,1,0,0,1,0),(0,1,0,0,1,1,0,0,1),(0,0,-1,-1,0,-1,-1,0,0),(-1,0,0,-1,-1,0,0,-1,0),(0,-1,0,0,-1,-1,0,0,1),(0,0,-1,0,1,-1,0,1,0),(0,1,0,-1,1,0,-1,0,0),(-1,0,0,0,0,1,-1,0,1)]
		del_E_l+=flip_spin*(lattice[mod1(start_i+i,L),mod1(start_j+j,L),mod1(start_k+k,L)]*lattice[mod1(start_i+l,L),mod1(start_j+m,L),mod1(start_k+n,L)]*lattice[mod1(start_i+p,L),mod1(start_j+q,L),mod1(start_k+r,L)])
	end
	del_E=(2.0*constant_j*del_E_j)+(2.0*constant_k*del_E_k)+(2.0*constant_l*del_E_l)
	if del_E<=0.0
		lattice[start_i,start_j,start_k]*=-1
	elseif rand()<exp(-del_E/T)
		lattice[start_i,start_j,start_k]*=-1
	end
	return lattice
end

function correlations(L::Int64,lattice::Array{Int64,3})::Array{Float64,2}
	if L%2==0
		even_dim=convert(Int64,((L/2)^3+6))
		correlations_list=zeros(Float64,even_dim,2)
		count=0
		for p=0:convert(Int64,((L/2)-1)),q=0:convert(Int64,((L/2)-1)),r=0:convert(Int64,((L/2)-1))
			dist=0
			correlation=0
			dist=distance(p,q,r)
			if dist>=1.0
				count+=1
				for i=1:L,j=1:L,k=1:L
					correlation+=lattice[i,j,k]*lattice[mod1(i+p,L),mod1(j+q,L),mod1(k+r,L)]
				end
				correlation/=float(L*L*L)
				correlations_list[count,1]=dist
				correlations_list[count,2]=correlation
			end
		end
		for p=[0,convert(Int64,L/2)],q=[0,convert(Int64,L/2)],r=[0,convert(Int64,L/2)]
			dist=0
			correlation=0
			dist=distance(p,q,r)
			if dist>=1.0
				count+=1
				for i=1:Int(L/2),j=1:Int(L/2),k=1:Int(L/2)
					correlation+=lattice[i,j,k]*lattice[mod1(i+p,L),mod1(j+q,L),mod1(k+r,L)]
				end
				correlation/=float(L*L*L)
				correlations_list[count,1]=dist
				correlations_list[count,2]=correlation
			end
		end

	else
		odd_dim=convert(Int64,(((L-1)/2)+1)^3-1)
		correlations_list=zeros(Float64,odd_dim,2)
		count=0
		for p=0:convert(Int64,((L-1)/2)),q=0:convert(Int64,((L-1)/2)),r=0:convert(Int64,((L-1)/2))
			dist=0
			correlation=0
			dist=distance(p,q,r)
			if dist>=1.0
				count+=1
				for i=1:L,j=1:L,k=1:L
					correlation=lattice[i,j,k]*lattice[mod1(i+p,L),mod1(j+q,L),mod1(k+r,L)]
				end
				correlation/=float(L*L*L)
				correlations_list[count,1]=dist
				correlations_list[count,2]=correlation
			end
		end
	end
	return correlations_list
end

function calc_energy(L::Int64,kappa::Float64,lattice::Array{Int64,3})::Float64
	energy,C1,C2,Ck=0.0,0.0,0.0,0.0
	constant_j,constant_k,constant_l=2*kappa,-kappa/2.0,(1-kappa)/2.0
	for i=1:L,j=1:L,k=1:L
		C1 +=lattice[i,j,k]*(lattice[mod1(i+1,L),j,k]+lattice[i,mod1(j+1,L),k]+lattice[i,j,mod1(k+1,L)])
		for (p,q,r) in [(0,1,1),(0,1,-1),(1,1,0),(1,-1,0),(1,0,1),(1,0,-1)]
			C2+=lattice[i,j,k]*lattice[mod1(i+p,L),mod1(j+q,L),mod1(k+r,L)]
		end
		for (l,m,n,p,q,r,x,y,z) in [(0,0,-1,1,0,-1,1,0,0),(1,0,0,0,-1,0,1,-1,0),(0,0,-1,0,-1,-1,0,-1,0)]
			Ck+=lattice[i,j,k]*lattice[mod1(i+p,L),mod1(j+q,L),mod1(k+r,L)]*lattice[mod1(i+l,L),mod1(j+m,L),mod1(k+n,L)]*lattice[mod1(i+x,L),mod1(j+y,L),mod1(k+z,L)]
		end
	end
	C1 /=float(L*L*L)
	C2 /=float(L*L*L)
	Ck /=float(L*L*L)
	energy = -constant_j*C1-constant_k*C2-constant_l*Ck
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

function energy_binder_simulation(L::Int64,kappa::Float64,T::Float64,sweeps::Int64)
	one_sweep=L*L*L
	mkpath("/scratch/fermi/jawla/Ongoing/binder_data_L=$(L)/kappa=$(kappa)_T=$(T)")
	lattice=initial_lattice(L,cold_start=false)
	open("/scratch/fermi/jawla/Ongoing/binder_data_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt","w") do f
		for i in 1:sweeps
			for step in 1:one_sweep
				lattice_1=metropolis_step(L,kappa,T,lattice)
				lattice=lattice_1
			end
			energy=calc_energy(L,kappa,lattice)
			write(f,"$(energy)\n")
		end
	end
	E_path="/scratch/fermi/jawla/Ongoing/binder_data_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt"
	array=Matrix(CSV.read(E_path,header=false))
	array=reshape(array,length(array))
	sweeps=length(array)
	tau=autocorrelation(array)
	energy_list=Float64[]
	open("/scratch/fermi/jawla/Ongoing/binder_data_L=$(L)/kappa=$(kappa)_T=$(T)/tau.txt","w") do f
		write(f,"$(tau)")
	end
	for i in collect(range(10*tau,sweeps,step=2*tau))
		push!(energy_list,array[i])
	end
	energy_4th=energy_list.^4
	energy_2nd=energy_list.^2
	avg_energy_4th=mean(energy_4th)
	avg_energy_2nd=mean(energy_2nd)
	avg_energy=mean(energy_list)
	binder_energy=1-(avg_energy_4th/(3*avg_energy_2nd^2))
	open("/scratch/fermi/jawla/Ongoing/binder_data_L=$(L)/kappa=$(kappa)_T=$(T)/binder_parameter.txt","w") do f
		write(f,"$(binder_energy)")
	end
	mkpath("/scratch/fermi/jawla/Data/binder_data_L=$(L)/kappa=$(kappa)_T=$(T)")
	mv("/scratch/fermi/jawla/Ongoing/binder_data_L=$(L)/kappa=$(kappa)_T=$(T)","/scratch/fermi/jawla/Data/binder_data_L=$(L)/kappa=$(kappa)_T=$(T)",force=true)
	return nothing
end

function phases_simulation(L::Int64,kappa::Float64,T::Float64,sweeps::Int64)
	one_sweep=L*L*L
	mkpath("/scratch/fermi/jawla/Ongoing/clustering_data_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data")
	lattice=initial_lattice(L,cold_start=false)
	open("/scratch/fermi/jawla/Ongoing/clustering_data_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt","w") do f
		for i in 1:sweeps
			for step in 1:one_sweep
				lattice_1=metropolis_step(L,kappa,T,lattice)
				lattice=lattice_1
			end
			corr_list=correlations(L,lattice)
			energy=calc_energy(L,kappa,lattice)
			write(f,"$(energy)\n")
			if i>= 20000 && i%1000==0
				open("/scratch/fermi/jawla/Ongoing/clustering_data_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data/sweep=$(i).txt", "w") do io
					for i in eachrow(corr_list)
						write(io,"$(i)\n")
					end
				end
			end
		end
	end


	return nothing
end

function continue_or_not(L::Int64,kappa::Float64,T::Float64,sweeps::Int64,lattice::Array{Int64,3})
	E_path="/scratch/fermi/jawla/Ongoing/clustering_data_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt"
	array=Matrix(CSV.read(E_path,header=false))
	array=reshape(array,length(array))
	sweeps=length(array)
	tau=autocorrelation(array)
	open("/scratch/fermi/jawla/Ongoing/clustering_data_L=$(L)/kappa=$(kappa)_T=$(T)/tau.txt","w") do f
		write(f,"$(tau)")
	end
	if 25*tau<=sweeps
		mkpath("/scratch/fermi/jawla/Data/clustering_data_L=$(L)/kappa=$(kappa)_T=$(T)")
		mv("/scratch/fermi/jawla/Ongoing/clustering_data_L=$(L)/kappa=$(kappa)_T=$(T)","/scratch/fermi/jawla/Data/clustering_data_L=$(L)/kappa=$(kappa)_T=$(T)",force=true)
	else
		new_sweeps=30*tau
		continue_phases_simulation(L,kappa,T,new_sweeps,sweeps,lattice)
	end
end

function continue_phases_simulation(L::Int64,kappa::Float64,T::Float64,extra_sweeps::Int64,sweeps::Int64,lattice::Array{Int64,3})
	one_sweep=L*L*L
	open("/scratch/fermi/jawla/Ongoing/clustering_data_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt","a") do f
		for i in sweeps+1:extra_sweeps
			for step in 1:one_sweep
				lattice_1=metropolis_step(L,kappa,T,lattice)
				lattice=lattice_1
			end
			corr_list=correlations(L,lattice)
			energy=calc_energy(L,kappa,lattice)
			write(f,"$(energy)\n")
			if i>= 20000 && i%1000==0
				open("/scratch/fermi/jawla/Ongoing/clustering_data_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data/sweep=$(i).txt", "w") do io
					for i in eachrow(corr_list)
						write(io,"$(i)\n")
					end
				end
			end
		end
	end
	sweeps=extra_sweeps
	continue_or_not(L,kappa,T,sweeps,lattice)
	return nothing
end

function dnn_simulation(L::Int64,kappa::Float64,T::Float64,sweeps::Int64)
	one_sweep=L*L*L
	mkpath("/scratch/fermi/jawla/Ongoing/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data")
	lattice=initial_lattice(L,cold_start=false)
	open("/scratch/fermi/jawla/Ongoing/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt","w") do f
		for i in 1:sweeps
			for step in 1:one_sweep
				lattice_1=metropolis_step(L,kappa,T,lattice)
				lattice=lattice_1
			end
			corr_list=correlations(L,lattice)
			energy=calc_energy(L,kappa,lattice)
			write(f,"$(energy)\n")
			if i>= 20000 && i%1000==0
				open("/scratch/fermi/jawla/Ongoing/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data/sweep=$(i).txt", "w") do io
					for i in eachrow(corr_list)
						write(io,"$(i)\n")
					end
				end
			end
		end
	end
	continue_dnn_or_not(L,kappa,T,sweeps,lattice)
	return nothing
end

function continue_dnn_or_not(L::Int64,kappa::Float64,T::Float64,sweeps::Int64,lattice::Array{Int64,3})
	E_path="/scratch/fermi/jawla/Ongoing/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt"
	array=Matrix(CSV.read(E_path,header=false))
	array=reshape(array,length(array))
	sweeps=length(array)
	tau=autocorrelation(array)
	open("/scratch/fermi/jawla/Ongoing/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)/tau.txt","w") do f
		write(f,"$(tau)")
	end
	if 25*tau<=sweeps
		mkpath("/scratch/fermi/jawla/Data/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)")
		mv("/scratch/fermi/jawla/Ongoing/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)","/scratch/fermi/jawla/Data/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)",force=true)
	else
		new_sweeps=30*tau
		continue_dnn_simulation(L,kappa,T,new_sweeps,sweeps,lattice)
	end
end

function continue_dnn_simulation(L::Int64,kappa::Float64,T::Float64,extra_sweeps::Int64,sweeps::Int64,lattice::Array{Int64,3})
	one_sweep=L*L*L
	open("/scratch/fermi/jawla/Ongoing/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt","a") do f
		for i in sweeps+1:extra_sweeps
			for step in 1:one_sweep
				lattice_1=metropolis_step(L,kappa,T,lattice)
				lattice=lattice_1
			end
			corr_list=correlations(L,lattice)
			energy=calc_energy(L,kappa,lattice)
			write(f,"$(energy)\n")
			if i>= 20000 && i%1000==0
				open("/scratch/fermi/jawla/Ongoing/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data/sweep=$(i).txt", "w") do io
					for i in eachrow(corr_list)
						write(io,"$(i)\n")
					end
				end
			end
		end
	end
	sweeps=extra_sweeps
	continue_dnn_or_not(L,kappa,T,sweeps,lattice)
	return nothing
end
end
kappa=[1.5]
lattices=[16]
temp1=collect(range(0.0,stop=1.65, step=0.033))
temp2=collect(range(3.0,stop=4.65, step=0.033))
temp=vcat(temp1,temp2)
comb=[[a,b,c] for a in lattices,b in kappa, c in temp]
z=reshape(comb,length(comb))
pmap(dnn_simulation,[Int(i[1]) for i in z],[i[2] for i in z],[i[3] for i in z],repeat([100000],length(comb)))
