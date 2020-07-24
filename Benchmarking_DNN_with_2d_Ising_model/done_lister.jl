using DelimitedFiles
function lister()
kappa=[1.5]
lattices=[16]
temp1=collect(range(0.0,stop=1.617, step=0.033))
temp2=collect(range(3.0,stop=4.617, step=0.033))
temp=vcat(temp1,temp2)
comb=[[a,b,c] for a in lattices,b in kappa, c in temp]
z=reshape(comb,length(comb))
new_list=Array{Float64,1}[]
for i in 1:length(z)
	L=Int(z[i][1])
	kappa=z[i][2]
	T=z[i][3]
	if ispath("/scratch/fermi/jawla/Data/dnn_data_L=$(L)/kappa=$(kappa)_T=$(T)")
		push!(new_list,comb[i])

	else
		println(false)
	end
end

writedlm("/home/jawla/Simulations/1/done_list.txt",new_list)
end
lister()
