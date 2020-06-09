using CSV, DataFrames
using DelimitedFiles

z=readdlm("/home/jawla/Simulations/done_list.txt")
corr_data=DataFrame()
for i in length(z)
    L=z[i][1]
	kappa=z[i][2]
	T=z[i][3]
	tau=readdlm("/scratch/fermi/jawla/Data/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)/tau.txt")
	energy_series=readdlm("/scratch/fermi/jawla/Data/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)/energy_timeseries.txt",Float64)
	sweeps=length(energy_series)
	for i in collect(range(1000000,sweeps,step=100))
		energy=energy_series[i]
		df=CSV.read("/scratch/fermi/jawla/Data/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data/sweep=$(i).txt",header=false)
		df=DataFrame(permutedims(Matrix(df)))
		rename!(df,[Symbol(i) for i in Vector(df[1,:])])
		df[!,:energy]=[energy]
		df[!,:temperature]=[T]
		df[!,:autocorrelation]=[tau]
		if T<=1.65
			df[!,:phase]=[0]
		elseif
			T>=3.0
			df[!,:phase]=[1]
		end
		r=vcat(corr_data,df)
		corr_data=r
	end
end
CSV.write("/home/jawla/Simulations/DNN_gonihedric_data.csv",corr_data)
