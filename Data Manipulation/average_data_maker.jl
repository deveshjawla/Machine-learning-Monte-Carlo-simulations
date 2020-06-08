using CSV, DataFrames
using DelimitedFiles
using Statistics
function data_maker()
	kappa=1.5
	temp1=collect(range(0.0,stop=1.5, step=0.0001))
	temp2=collect(range(3.0,stop=4.5, step=0.0001))
	temp=vcat(temp1,temp2)
    # temp=collect(range(1.72,stop=3.21, step=0.015))
    # kappa=collect(range(0.0,stop=1.0, step=0.01))
    # comb=[[aa,bb] for aa in kappa,bb in temp]
    # comb=reshape(comb,(length(kappa)*length(temp)))
    # new_list=Array{Float64,1}[]
	# tau_list=Float64[]
    # for i in 1:length(comb)
    # 	kappa=comb[i][1]
    # 	T=comb[i][2]
    # 	if ispath("/home/jawla/jobs/data_pairwise_correlations_L=16/kappa=$(kappa)_T=$(T)")
    # 		push!(new_list,comb[i])
    # 	else
    # 		println(false)
    # 	end
    # end
    # comb=new_list
	corr_data=DataFrame()
	for i in length(temp)
		T=temp[i]
	# for i in 1:size(comb,1)
		# kappa=comb[i][1]
		# T=comb[i][2]
		tau=readdlm("/home/jawla/jobs/data_pairwise_correlations_L=16/kappa=$(kappa)_T=$(T)/tau.txt")
		E_path="/scratch/fermi/jawla/data_pairwise_correlations_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt"
		array=readdlm(E_path,Float64)
		sweeps=length(array)
		energy_list=Float64[]
		for i in collect(range(5*tau,sweeps,step=2*tau))
			push!(energy_list,array[i])
		end
		energy_4th=energy_list.^4
		energy_2nd=energy_list.^2
		avg_energy_4th=mean(energy_4th)
		avg_energy_2nd=mean(energy_2nd)
		avg_energy=mean(energy_list)
		binder_energy=1-(avg_energy_4th/(3*avg_energy_2nd^2))
		# push!(tau_list,tau)
		df=CSV.read("/home/jawla/jobs/data_pairwise_correlations_L=16/kappa=$(kappa)_T=$(T)/average_correlation.csv")
		df=DataFrame(permutedims(Matrix(df)))
		rename!(df,[Symbol(i) for i in Vector(df[1,:])])
		select!(df, Not(1))
		df=mapcols(prod,df)
		df[!,:energy]=[avg_energy]
		df[!,:binder_energy_cumulant]=[binder_energy]
		# df[!,:kappa]=[kappa]
		df[!,:temperature]=[T]
		if T<=1.5
			df[!,:phase]=[0]
		elseif
			T>=3.0
			df[!,:phase]=[1]
		end
		r=vcat(corr_data,df)
		corr_data=r
	end
	# writedlm("tau_list.txt",tau_list)
	CSV.write("/home/jawla/jobs/DNN_gonihedric_data.csv",corr_data)
	return corr_data
end

df=data_maker()
