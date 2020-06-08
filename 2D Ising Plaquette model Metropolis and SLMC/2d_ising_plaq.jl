using Random

function initial_lattice(L::Int64;cold_start=true)::Array{Int64,2}
    lattice=ones(Int64,L,L)
    if cold_start==false
        lattice=rand([1,-1],L,L)
    end
    return lattice
end
function metropolis_step(L::Int64,T::Float64,lattice::Array{Int64,2})::Array{Int64,2}
    C_1_a=0.0
    C_2=0.0
    C_k_a=0.0
    C_1_b=0.0
    C_2=0.0
    C_k_b=0.0
    del_E=0.0
    mag=0
    eng=0
    E_a=0
    E_a_eff=0
    E_b=0
    E_b_eff=0
    constant_j=1.0
    constant_k=0.2*constant_j
    start_i,start_j=rand(1:L,1,2)
    flip_spin=lattice[start_i,start_j]
    cluster = [(start_i,start_j)]
    done = np.zeros((L,L))
    N = 0
    while N < len(cluster):
        for (i,j) in [(1,0),(0,1),(-1,0),(0,-1)]:
            #mod L for periodic boundary
            neighbor = (mod1(cluster[N][0]+i,L),mod1(cluster[N][1]+j,L))
            #print(neighbor)
            if lattice[neighbor] == seed_spin:
                #if neighbor not in cluster:
                if done[neighbor] != 1:
                    if rand()<=(1.0-exponent):
                        cluster.append(neighbor)
                        done[neighbor] = 1
                        #print(N,len(cluster))
                    end
                end
            end
        end
        N += 1

    return lattice
end

        #calcultate the energy of the configuration before flipping
        for i in range(L):
            for j in range(L):
                C_1_a+= lattice[i,j]*(lattice[(i+1)%L, j] + lattice[i,(j-1)%L])
                #C_2+= lattice[i,j]*(lattice[(i+1)%L, (j+1)%L] + lattice[(i+1)%L, (j-1)%L])
                C_k_a+= lattice[i,j]*(lattice[(i+1)%L, j]*lattice[i, (j-1)%L]*lattice[(i+1)%L, (j-1)%L])
        C_1_a/=float(L*L)
        #C_2/=float(L*L)
        C_k_a/=float(L*L)
        E_a= -constant_j*C_1_a-constant_k*C_k_a
        E_a_eff= 0.0124-1.0867*C_1_a
        #we flip the cluster
        for i in range(len(cluster)):
            lattice[cluster[i][0]][cluster[i][1]] *= -1
        for i in range(L):
            for j in range(L):
                C_1_b+= lattice[i,j]*(lattice[(i+1)%L, j] + lattice[i,(j-1)%L])
                #C_2+= lattice[i,j]*(lattice[(i+1)%L, (j+1)%L] + lattice[(i+1)%L, (j-1)%L])
                C_k_b+= lattice[i,j]*(lattice[(i+1)%L, j]*lattice[i, (j-1)%L]*lattice[(i+1)%L, (j-1)%L])
        C_1_b/=float(L*L)
        #C_2/=float(L*L)
        C_k_b/=float(L*L)
        E_b= -constant_j*C_1_b-constant_k*C_k_b
        E_b_eff= 0.0124-1.0867*C_1_b
        del_E=(E_b-E_b_eff)-(E_a-E_a_eff)
        #magnetisation per iteration
#         for i in range(L):
#                 for j in range(L):
#                     mag +=lattice[i][j]
#         mag_timeseries.append(np.abs(mag))
        eng=E_b
        C_1=C_1_b
        rand2=random.SystemRandom().random()
        #here instead of using the acceptance ration, we use the rejection ratio of the flipped cluster
        #this leads us to backflip the cluster
        if rand2>np.exp(-del_E/T):
            for i in range(len(cluster)):
                lattice[cluster[i][0]][cluster[i][1]] *= -1
            eng=E_a
            C_1=C_1_a
        energy_timeseries.append(eng)
        C_1_timeseries.append(C_1)
#     running average magnetisation
#     list_avg_mag=[]
#     for i in range(len(mag_timeseries)):
#         avgmag=0
#         for j in range(i+1):
#             avgmag+=mag_timeseries[j]
#         list_avg_mag.append(avgmag/(i+1))
#     saving files
#     lattice_config_file = "wolff_data/" + "config_" + str(T) + "_L" + str(L)
#     np.save(lattice_config_file,lattice)
#     mag_timeseries_file = "wolff_data/" + "mag_timeseries_" + str(T) + "_L" + str(L) + ".txt"
#     np.savetxt(mag_timeseries_file,mag_timeseries)
    energy_timeseries_file = "wolff_data/" + "energy_timeseries_SLMC_2d_plaquette_periodic" + str(T) + "_L" + str(L) + ".txt"
    np.savetxt(energy_timeseries_file,energy_timeseries)
    C_1_timeseries_file = "wolff_data/" + "C_1_timeseries_SLMC_2d_plaquette_periodic" + str(T) + "_L" + str(L) + ".txt"
    np.savetxt(C_1_timeseries_file,C_1_timeseries)
    #plotting observables
    plt.plot(energy_timeseries,label="Wolff"+str(T)+"_"+str(L))
    plt.legend()
    return None

    temp_list=np.linspace(2.493,6,1)
temp_expo_list=np.exp(-2.0*1.0867/temp_list)
#function executing wolff-algorithm for different temperature and L; saving results as "mag_timeseries_Txxx_Lxxx.txt"
def do_and_save():
    L_start = 20
    L_end = 20
    L_step = 20
    L_count = 1+int((L_end-L_start)/L_step)

    CPU_timematrix = np.zeros((len(temp_list),L_count))

    #loop over L's
    for i in range(len(temp_list)):
        T = temp_list[i]
        expo=temp_expo_list[i]

        for j in range(L_count):
            L = L_start + j * L_step
            t_in = time.time()
            cluster_update(T,L,50000,expo)
            t_fin = time.time()
            CPU_timematrix[i][j] = t_fin - t_in
            print("done L = " + str(L) + ", T = " + str(T))



    np.savetxt("CPU_times/cputimes3_"+ "SLMC_2d_plaquette_periodic, T = " +str(T)+"L_"+str(L)+".txt",CPU_timematrix)
do_and_save()
