using Random
using Distributed
addprocs(40) # change as needed
size=[8,12,16]
temp=collect(range(4.49,stop=4.54,length=10))
comb=[[aa,bb] for aa in size,bb in temp]
comb=reshape(comb,(1,length(size)*length(temp)))

@sync @distributed for i in 1:length(comb)
    function initial_lattice(L::Int64;cold_start=true)::Array{Int64,3}
        lattice=ones(Int64,L,L,L)
        if cold_start==false
            lattice=rand([1,-1],L,L,L)
        end
        return lattice
    end
    
    function metropolis_step(L::Int64,T::Float64,lattice::Array{Int64,3})::Array{Int64,3}
        del_E_j=0.0
        del_E=0.0
        constant_j=1.0
        start_i,start_j,start_k=rand(1:L,1,3)
        flip_spin=lattice[start_i,start_j,start_k]
        for (i,j,k) in [(1,0,0),(0,1,0),(0,0,1),(-1,0,0),(0,-1,0),(0,0,-1)]
            del_E_j+= flip_spin*lattice[mod1(start_i+i,L),mod1(start_j+j,L),mod1(start_k+k,L)]
        end
        del_E=2.0*constant_j*del_E_j
        if del_E<=0.0
            lattice[start_i,start_j,start_k]*=-1
        elseif rand()<exp(-del_E/T)
            lattice[start_i,start_j,start_k]*=-1
        end
        return lattice
    end

    function calc_mag_energy(L::Int64,T::Float64,lattice::Array{Int64,3},one_sweep::Int64)#return magnetisation after one sweep
        mag=0.0
        energy=0.0
        C1=0.0
        constant_j=1.0
        for i in 1:one_sweep
            lattice_1=metropolis_step(L,T,lattice)
            lattice=lattice_1
        end
        for i=1:L,j=1:L,k=1:L
            mag +=lattice[i,j,k]
            C1 +=lattice[i,j,k]*(lattice[mod1(i+1,L),j,k]+lattice[i,mod1(j+1,L),k]+lattice[i,j,mod1(k+1,L)])
        end
        mag/=float(L*L*L)
        C1 /=float(L*L*L)
        energy = -constant_j*C1
        return C1,energy,mag,lattice
    end


    function simulation(L::Int64,T::Float64,sweeps::Int64)

        one_sweep=L*L*L
        lattice=initial_lattice(L,cold_start=false)
        open("3d_plaq_mag_energy_timeseries_T=$(T)_L=$(L).txt","w") do f
            for i in 1:sweeps
                C1_2,energy_2,mag_2,lattice_2=calc_mag_energy(L,T,lattice,one_sweep)
                mag=mag_2
                lattice=lattice_2
                energy=energy_2
                if i%10==0
                    write(f,"$(mag)\t$(energy)\n")
                end
            end
        end
    end
    ii=convert(Int64,comb[i][1])
    jj=comb[i][2]

    println(ii," - ",jj)
    simulation(ii,jj,1000000)
end
