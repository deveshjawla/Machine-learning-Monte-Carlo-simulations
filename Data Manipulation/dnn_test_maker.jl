using CSV, DataFrames
using DelimitedFiles

z=readdlm("/home/jawla/Simulations/done_list.txt")
corr_data=DataFrame()
for i in length(z)
    L=z[i][1]
	kappa=z[i][2]
	T=z[i][3]
	df=CSV.read("/scratch/fermi/jawla/Data/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)/sweep_data/sweep=3000000.txt",header=false)
	df=DataFrame(permutedims(Matrix(df)))
	rename!(df,[Symbol(i) for i in Vector(df[1,:])])
	df[!,:temperature]=[T]
	r=vcat(corr_data,df)
	corr_data=r
end
CSV.write("/home/jawla/Simulations/DNN_gonihedric_test.csv",corr_data)
