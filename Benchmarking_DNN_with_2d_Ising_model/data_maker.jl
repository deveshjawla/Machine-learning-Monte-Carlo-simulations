using DelimitedFiles
function data_maker()
z=readdlm("/home/jawla/Simulations/1/done_list.txt")
y=readdlm("/home/jawla/Simulations/1/notdone_list.txt")
for i in 1:size(z)[1]
    L=Int(z[i,1])
	T=z[i,2]
	tau=readdlm("/scratch/fermi/jawla/Data/2d_ising_dnn_data_L=$(L)_T=$(T)/tau.txt")[1]
	energy_series=readdlm("/scratch/fermi/jawla/Data/2d_ising_dnn_data_L=$(L)_T=$(T)/energy_timeseries.txt",Float64)
	sweeps=length(energy_series)
	for i in collect(range(1000000,sweeps,step=100))
		energy=energy_series[i]
		df=readdlm("/scratch/fermi/jawla/Data/2d_ising_dnn_data_L=$(L)_T=$(T)/sweep_data/sweep=$(i).txt",',',Float64)[:,2]
		push!(df,T)
		if T<=1.65
			push!(df,0)
		elseif T>=3.0
			push!(df,1)
		end
		df=permutedims(df)
		open(io-> writedlm(io,df,','),"/scratch/fermi/jawla/2d_ising_dnn_data.csv","a")
	end
end
for i in 1:size(y)[1]
    L=Int(y[i,1])
	T=y[i,2]
	tau=readdlm("/scratch/fermi/jawla/Ongoing/2d_ising_dnn_data_L=$(L)_T=$(T)/tau.txt")[1]
	energy_series=readdlm("/scratch/fermi/jawla/Ongoing/2d_ising_dnn_data_L=$(L)_T=$(T)/energy_timeseries.txt",Float64)
	sweeps=length(energy_series)
	for i in collect(range(1000000,sweeps,step=100))
		energy=energy_series[i]
		df=readdlm("/scratch/fermi/jawla/Ongoing/2d_ising_dnn_data_L=$(L)_T=$(T)/sweep_data/sweep=$(i).txt",',',Float64)[:,2]
		push!(df,T)
		if T<=1.65
			push!(df,0)
		elseif T>=3.0
			push!(df,1)
		end
		df=permutedims(df)
		open(io-> writedlm(io,df,','),"/scratch/fermi/jawla/2d_ising_dnn_data.csv","a")
	end
end
end
@time begin
data_maker()
end
