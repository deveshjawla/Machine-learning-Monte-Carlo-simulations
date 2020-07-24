using CSV
using Random
using Printf
function to_bin()
	df=Float32[]
   	for row in CSV.Rows("/scratch/fermi/jawla/2d_ising_dnn_data.csv",delim=',',header=false,reusebuffer=true)
   		for i in 1:522
			push!(df,parse(Float32,row[i]))
    		end
    	end
    len=4220100
	df=reshape(df,(522,len))
	df=permutedims(df)
	nanrows=any(isnan.(df),dims=2)
	df=df[.!vec(nanrows),:]
	len=size(df)[1]
    	df=df[shuffle(1:len),:]
	@info(@sprintf("%d is the size of the array",len))
	write("/scratch/fermi/jawla/gonihedric_training_data.bin",df)
end
to_bin()
