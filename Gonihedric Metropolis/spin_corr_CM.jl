using Distributed
using ClusterManagers
addprocs(SlurmManager(100))
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere println(pwd())
@everywhere using FFTW
@everywhere using Random
@everywhere using Pandas
@everywhere using DelimitedFiles
@everywhere using Statistics
@everywhere using LsqFit
kappa=1.5
temp1=collect(range(0.0,stop=1.5, step=0.0001))
temp2=collect(range(3.0,stop=4.5, step=0.0001))
temp=vcat(temp1,temp2)
# temp=collect(range(1.72,stop=3.21, step=0.015))
# kappa=collect(range(0.0,stop=1.0, step=0.01))
# comb=[[aa,bb] for aa in kappa,bb in temp]
# comb=reshape(comb,(length(kappa)*length(temp)))
# new_list=Array{Float64,1}[]
# for i in 1:length(comb)
# 	kappa=comb[i][1]
# 	T=comb[i][2]
# 	if ispath("/home/jawla/jobs/data_pairwise_correlations_L=16/kappa=$(kappa)_T=$(T)")
# 		println(false)
# 	else
# 		push!(new_list,comb[i])
# 	end
# end
# writedlm("/home/jawla/jobs/list.txt",new_list)
# comb=readdlm("/home/jawla/jobs/list.txt")
# @sync @distributed for i in 1:size(comb,1)
# @sync @distributed for i in 1:length(comb)
@sync @distributed for i in 1:length(temp)
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
		del_E_j=0.0
		del_E_k=0.0
		del_E_l=0.0
		del_E=0.0
		constant_j=2*kappa
		constant_k=-kappa/2.0
		constant_l=(1-kappa)/2.0
		start_i,start_j,start_k=rand(1:L,1,3)
		flip_spin=lattice[start_i,start_j,start_k]
		for (i,j,k) in [(1,0,0),(0,1,0),(0,0,1),(-1,0,0),(0,-1,0),(0,0,-1)]
			del_E_j+= flip_spin*lattice[mod1(start_i+i,L),mod1(start_j+j,L),mod1(start_k+k,L)]
		end
		for (i,j,k) in [(1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),(-1,1,0),(1,1,0),(1,-1,0),(-1,-1,0),(0,-1,1),(0,-1,-1),(0,1,-1),(0,1,1)]
			del_E_k+= flip_spin*lattice[mod1(start_i+i,L),mod1(start_j+j,L),mod1(start_k+k,L)]
		end
		del_E_l=flip_spin*((lattice[start_i,start_j,mod1(start_k-1,L)]*lattice[mod1(start_i+1,L),start_j,mod1(start_k-1,L)]*lattice[mod1(start_i+1,L),start_j,start_k])+(lattice[mod1(start_i+1,L),start_j,start_k]*lattice[start_i,mod1(start_j-1,L),start_k]*lattice[mod1(start_i+1,L),mod1(start_j-1,L),start_k])+(lattice[start_i,start_j,mod1(start_k-1,L)]*lattice[start_i,mod1(start_j-1,L),mod1(start_k-1,L)]*lattice[start_i,mod1(start_j-1,L),start_k])+(lattice[start_i,start_j,mod1(start_k+1,L)]*lattice[mod1(start_i+1,L),start_j,mod1(start_k+1,L)]*lattice[mod1(start_i+1,L),start_j,start_k])+(lattice[mod1(start_i+1,L),start_j,start_k]*lattice[mod1(start_i,L),mod1(start_j,L),start_k]*lattice[start_i,mod1(start_j+1,L),start_k])+(lattice[start_i,mod1(start_j+1,L),start_k]*lattice[start_i,mod1(start_j+1,L),mod1(start_k+1,L)]*lattice[start_i,start_j,mod1(start_k+1,L)])+(lattice[start_i,start_j,mod1(start_k,L)]*lattice[mod1(start_i-1,L),start_j,mod1(start_k+1,L)]*lattice[mod1(start_i-1,L),start_j,start_k])+(lattice[mod1(start_i-1,L),start_j,start_k]*lattice[mod1(start_i-1,L),mod1(start_j-1,L),start_k]*lattice[start_i,mod1(start_j-1,L),start_k])+(lattice[start_i,mod1(start_j-1,L),start_k]*lattice[mod1(start_i-1,L),start_j,mod1(start_k-1,L)]*lattice[start_i,start_j,mod1(start_k+1,L)])+(lattice[mod1(start_i-1,L),start_j,start_k]*lattice[mod1(start_i-1,L),start_j,mod1(start_k-1,L)]*lattice[start_i,start_j,mod1(start_k-1,L)])+(lattice[start_i,start_j,mod1(start_k-1,L)]*lattice[start_i,mod1(start_j+1,L),mod1(start_k-1,L)]*lattice[start_i,mod1(start_j+1,L),start_k])+(lattice[start_i,mod1(start_j+1,L),start_k]*lattice[mod1(start_i-1,L),mod1(start_j+1,L),start_k]*lattice[mod1(start_i-1,L),start_j,start_k]))
		del_E=(2.0*constant_j*del_E_j)+(2.0*constant_k*del_E_k)+(2.0*constant_l*del_E_l)
		#print(del_E_k,del_E,del_E_j)
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
				end
				if count>=1
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
					for i=1:L,j=1:L,k=1:L
						if i<=p
							correlation+=lattice[i,j,k]*lattice[mod1(i+p,L),mod1(j+q,L),mod1(k+r,L)]
						end
					end
					correlation/=float(L*L*L)
				end
				if count>=1
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
				end
				if count>=1
					correlations_list[count,1]=dist
					correlations_list[count,2]=correlation
				end
			end
		end
		return correlations_list
	end

	function calc_energy(L::Int64,lattice::Array{Int64,3})::Float64#return magnetisation after one sweep
		energy=0.0
		C1=0.0
		C2=0.0
		Ck=0.0
		kappa=1.5
		constant_j=2*kappa
		constant_k=-kappa/2.0
		constant_l=(1-kappa)/2.0
		for i=1:L,j=1:L,k=1:L
			C1 +=lattice[i,j,k]*(lattice[mod1(i+1,L),j,k]+lattice[i,mod1(j+1,L),k]+lattice[i,j,mod1(k+1,L)])
			C2 +=lattice[i,j,k]*(lattice[i,mod1(j+1,L),mod1(k+1,L)]+lattice[i,mod1(j+1,L),mod1(k-1,L)]+lattice[mod1(i+1,L),mod1(j+1,L),k]+lattice[mod1(i+1,L),mod1(j-1,L),k]+lattice[mod1(i+1,L),j,mod1(k+1,L)]+lattice[mod1(i+1,L),j,mod1(k-1,L)])
			Ck +=lattice[i,j,k]*((lattice[i,j,mod1(k-1,L)]*lattice[mod1(i+1,L),j,mod1(k-1,L)]*lattice[mod1(i+1,L),j,k])+(lattice[mod1(i+1,L),j,k]*lattice[i,mod1(j-1,L),k]*lattice[mod1(i+1,L),mod1(j-1,L),k])+(lattice[i,j,mod1(k-1,L)]*lattice[i,mod1(j-1,L),mod1(k-1,L)]*lattice[i,mod1(j-1,L),k]))
		end
		C1 /=float(L*L*L)
		C2 /=float(L*L*L)
		Ck /=float(L*L*L)
		energy = -constant_j*C1-constant_k*C2-constant_l*Ck
		return energy
	end

	# function tau(observable::Array{Float64,1})::Int64
	#     fm=FFTW.fft(observable[:].-mean(observable[:]))./sqrt(length(observable[:])) #fourier transform of mag
	#     fm2=(abs.(fm)).^2 #autocorrelation of the fourieer transformed mag
	#     cm=real(FFTW.ifft(fm2)) #inverse FT of the fm2, i.e. autocorrelation funciton of Observable(t)
	#     cm_2= cm ./ cm[1] #autocorrelation normalized
	#     log_cm_2 =[]#log list of autocorelation function(time-step)
	#     j=1
	#     while cm_2[j] > 0
	#         push!(log_cm_2,log(cm_2[j]))
	#         j = j+1
	#     end
	#     t = collect(LinRange(0,length(log_cm_2)-1,length(log_cm_2)))
	#     @. line(t,p)=p[1]*t+p[2]
	#     p0=[0.5,0.5]
	#     fit = curve_fit(line,t,log_cm_2,p0)
	#     p=fit.param
	#     tau = -1 / p[1]
	#     taue = 2*ceil(Int64,tau)
	#     correlation=10*ceil(Int64,2*taue/10)#making it in multiples of 10 because saving every 10th sweep
	#     return correlation
	# end

	function data_reducer(L::Int64,kappa::Float64,T::Float64,sweeps::Int64,taue::Int64)
		for i in collect(range(5*taue,sweeps,step=2*taue))
			df=read_csv("/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data/sweep=$(i).txt",header=nothing,names=["dist","corr"])
			df["dist"]=[x[2:end] for x in df["dist"]]
			df["dist"]=to_numeric(df["dist"],downcast="integer")
			df["corr"]=[y[1:end-1] for y in df["corr"]]
			df["corr"]=to_numeric(df["corr"],downcast="float")
			df["freq"] = transform(groupby(df,"dist")["dist"],"count")
			new_df=reset_index(mean(groupby(df,"dist")))
			to_csv(new_df,"/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data/sweep=$(i).csv",index=false)
		end

		df1=DataFrame(columns=["dist","corr","freq"])
		for i in collect(range(5*taue,sweeps,step=2*taue))
			df0 = concat([df1,read_csv("/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data/sweep=$(i).csv")])
			df1=df0
		end
		new_df1=reset_index(mean(groupby(df1,["dist","freq"])))
		to_csv(new_df1,"/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/average_correlation.csv",index=false)
		rm("/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data",recursive=true)
		return nothing
	end

	function dataframe_csv(L::Int64,kappa::Float64,T::Float64,E_path,sweeps::Int64,lattice::Array{Int64,3})
		array=readdlm(E_path,Float64)
		sweeps=length(array)
		observable = reshape(array,sweeps)
		function tau(observable::Array{Float64,1})::Int64
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
			taue = 2*ceil(Int64,tau)
			correlation=10*ceil(Int64,2*taue/10)#making it in multiples of 10 because saving every 10th sweep
			return correlation
		end
		taue = tau(observable)
		open("/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/tau.txt","w") do f
			write(f,"$(taue)")
		end
		if 25*taue<=sweeps
			data_reducer(L,kappa,T,sweeps,taue)
		else
			new_sweeps=30*taue
			continue_simulation(L,kappa,T,new_sweeps,sweeps,lattice)
		end
		return nothing
	end

	function continue_simulation(L::Int64,kappa::Float64,T::Float64,extra_sweeps::Int64,sweeps::Int64,lattice::Array{Int64,3})
		one_sweep=L*L*L
		open("/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt","a") do f
			for i in sweeps+1:extra_sweeps
				for step in 1:one_sweep
					lattice_1=metropolis_step(L,kappa,T,lattice)
					lattice=lattice_1
				end
				corr_list=correlations(L,lattice)
				energy=calc_energy(L,lattice)
				write(f,"$(energy)\n")
				if i%10==0
					open("/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data/sweep=$(i).txt", "w") do io
						for i in eachrow(corr_list)
							write(io,"$(i)\n")
						end
					end
				end
			end
		end
		sweeps=extra_sweeps
		E_path="/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt"
		dataframe_csv(L,kappa,T,E_path,sweeps,lattice)
		return nothing
	end


	function simulation(L::Int64,kappa::Float64,T::Float64,sweeps::Int64)
		one_sweep=L*L*L

			mkpath("/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data")
			lattice=initial_lattice(L,cold_start=false)
			open("/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt","w") do f
				for i in 1:sweeps
					for step in 1:one_sweep
						lattice_1=metropolis_step(L,kappa,T,lattice)
						lattice=lattice_1
					end
					corr_list=correlations(L,lattice)
					energy=calc_energy(L,lattice)
					write(f,"$(energy)\n")
					if i%10==0
						open("/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data/sweep=$(i).txt", "w") do io
							for i in eachrow(corr_list)
								write(io,"$(i)\n")
							end
						end
					end
				end
			end

		E_path="/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt"
		dataframe_csv(L,kappa,T,E_path,sweeps,lattice)
		return nothing
	end

	# ii=comb[i][1]
	# jj=comb[i][2]
	# ii=comb[i,1]
	# jj=comb[i,2]
	ii=kappa
	jj=temp[i]
	lattice_size=16
	simulation(lattice_size,ii,jj,10000)
end
rmprocs(workers())
