import numpy as np
import random
import time
from numba import jit
import math
import matplotlib.pyplot as plt
import secrets

def single_update(T,L,sweeps):
    one_sweep=L*L
    mag_timeseries=[]
    energy_timeseries=[]
    lattice=np.zeros([L,L])
    for i in range(L):
            for j in range(L):
                lattice[i][j] = secrets.choice([1,-1])
    for i in range(sweeps):
        mag=0.0
        eng=0.0
        for i in range(one_sweep):
            del_E=0.0
            rand=random.SystemRandom().random()
            start_i = secrets.randbelow(L)
            start_j = secrets.randbelow(L)
            flip_spin=lattice[start_i][start_j]
            for (i,j) in [(1,0),(0,1),(-1,0),(0,-1)]:
                del_E+= 2.0*(flip_spin*lattice[(start_i+i)%L][(start_j+j)%L])
            #print(del_E)
            if del_E<=0:
                lattice[start_i][start_j]*=-1
            elif rand<np.exp(-del_E/T):
                lattice[start_i][start_j]*=-1
        #instantaneous magnetisation after every 1 sweep
#         for i in range(L):
#                 for j in range(L):
#                     mag +=lattice[i][j]

#         mag_timeseries.append(np.abs(mag))
        #instantaneous energy after 1 sweep
        for i in range(L):
                for j in range(L):
                    eng+=-lattice[i,j]*(lattice[(i+1)%L, j] + lattice[i,(j+1)%L] + lattice[(i-1)%L, j] + lattice[i,(j-1)%L])/2.0
        energy_timeseries.append(eng/(float(L*L)))
    #saving files
    #lattice_config_file = "metropolis_data/" + "config_" + str(T) + "_L" + str(L) + "sweep no." + str(i)
    #np.save(lattice_config_file,lattice)
    #mag_timeseries_file = "metropolis_data/" + "mag_timeseries_" + str(T) + "_L" + str(L) + ".txt"
    #np.savetxt(mag_timeseries_file,mag_timeseries)
    energy_timeseries_file = "metropolis_data/" + "energy_timeseries_" + str(T) + "_L" + str(L) + ".txt"
    np.savetxt(energy_timeseries_file,energy_timeseries)

    #plotting observables
    #plt.plot(energy_timeseries,label="Metropolis"+str(T)+"_"+str(L))
    #plt.legend()
    return None


def cluster_update(T,L,iterations,exponent):
    lattice=np.zeros([L,L])
    #one_sweep=L*L/
    #setting up a random lattice, only once is suffiecnet since every state has one seed
    mag_timeseries=[]
    energy_timeseries=[]
    for i in range(L):
            for j in range(L):
                lattice[i][j] = 2.0*random.randint(0,1)-1

    #iteration loop
    for n in range(iterations):
        mag=0
        eng=0
        start_i = random.randint(0,L-1)
        start_j = random.randint(0,L-1)
        #to flip spinstate
        seed_spin = lattice[start_i][start_j]
        #list of to flip spins
        cluster = [(start_i,start_j)]
        #list of done neighbours
        done = np.zeros((L,L))
        #check neighbours
        N = 0
        while N < len(cluster):

            for (i,j) in [(1,0),(0,1),(-1,0),(0,-1)]:
                #mod L for periodic boundary
                neighbor = ((cluster[N][0]+i)%L,(cluster[N][1]+j)%L)
                #print(neighbor)

                if lattice[neighbor] == seed_spin:
                    #if neighbor not in cluster:
                    if done[neighbor] != 1:
                        rand=random.SystemRandom().random()
                        if rand<=(1.0-exponent):
                            cluster.append(neighbor)
                            done[neighbor] = 1
                            #print(N,len(cluster))

            N += 1
        #we flip the cluster
        for i in range(len(cluster)):
            lattice[cluster[i][0]][cluster[i][1]] *= -1

        #magnetisation per iteration

#         for i in range(L):
#                 for j in range(L):

#                     mag +=lattice[i][j]

#         mag_timeseries.append(np.abs(mag))

        for i in range(L):
                for j in range(L):
                    eng+=-lattice[i,j]*(lattice[(i+1)%L, j] + lattice[i,(j+1)%L] + lattice[(i-1)%L, j] + lattice[i,(j-1)%L])/2.0
        energy_timeseries.append(eng/float(L*L))
    #running average magnetisation
#     list_avg_mag=[]
#     for i in range(len(mag_timeseries)):
#         avgmag=0
#         for j in range(i+1):
#             avgmag+=mag_timeseries[j]
#         list_avg_mag.append(avgmag/(i+1))

    #saving files
#     lattice_config_file = "wolff_data/" + "config_" + str(T) + "_L" + str(L)
#     np.save(lattice_config_file,lattice)
#     mag_timeseries_file = "wolff_data/" + "mag_timeseries_" + str(T) + "_L" + str(L) + ".txt"
#     np.savetxt(mag_timeseries_file,mag_timeseries)
    energy_timeseries_file = "wolff_data/" + "energy_timeseries_" + str(T) + "_L" + str(L) + ".txt"
    np.savetxt(energy_timeseries_file,energy_timeseries)

    #plotting observables
#     plt.plot(energy_timeseries,label="Wolff"+str(T)+"_"+str(L))
#     plt.legend()
    return None

    exact_data=np.loadtxt("2d_is_e008.txt")
    beta_list=np.asarray([exact_data[i][0] for i in range(len(exact_data))])
    temp_list=1/beta_list
    #temp_list=np.linspace(1,5,1)
    temp_expo_list=np.exp(-2.0*beta_list)
    #function executing wolff-algorithm for different temperature and L; saving results as "maglist_Txxx_Lxxx.txt"
    def do_and_save():
        L_start = 8
        L_end = 8
        L_step = 20
        L_count = 1+int((L_end-L_start)/L_step)

        #CPU_timematrix = np.zeros((len(temp_list),L_count))

        #loop over L's
        for i in range(len(temp_list)):
            T = temp_list[i]
            expo=temp_expo_list[i]

            for j in range(L_count):
                L = L_start + j * L_step
                #t_in = time.time()
                cluster_update(T,L,2000,expo)
                #single_update(T,L,10000)
                #t_fin = time.time()
                #CPU_timematrix[i][j] = t_fin - t_in
                print("done L = " + str(L) + ", T = " + str(T))



        #np.savetxt("wolff_data/cputimes3.txt",CPU_timematrix)
    do_and_save()

    def tau(observable):


        fm=np.fft.rfft(observable[:]-np.mean(observable[:]))/np.sqrt(len(observable[:])) #fourier transform of mag

        fm2=np.abs(fm)**2 #autocorrelation of the fourieer transformed mag

        cm=np.fft.irfft(fm2, len(observable[:])) #inverse FT of the fm2, i.e. autocorrelation funciton of Observable(t)

        cm_2= cm / cm[0] #autocorrelation normalized

        #print(len(cm_2))

        log_cm_2 =[]#log list of autocorelation function(time-step)

        j=0

        while cm_2[j] > 0 :
            log_cm_2.append(np.log(cm_2[j]))
            j = j+1
        #print(len(log_cm_2))
        t = np.linspace(0,len(log_cm_2)-1,len(log_cm_2))
        p = np.polyfit(t,log_cm_2,1)
    #     a=len(log_cm_2)
    #     log_cm_2 = np.log(cm_2[:a/400 +1])
    #     x = np.linspace(0,a/400,a/400 +1)
    #     p = np.polyfit(x,log_cm_2,1)

        tau = -1 / p[0]

        correlation = 2*int(np.ceil(tau))
        corr_time = int(correlation)
        return corr_time
    tau_list=[]
    average_energy_list=[]
    for i in range(171):
        T=temp_list[i]
        name="wolff_data/" + "energy_timeseries_" + str(T) + "_L8.txt"

        energy_timeseries=np.loadtxt(name)
        #exact_energy_timeseries=np.asarray([exact_data[i][1] for i in range(len(exact_data))])
        tau_i=tau(energy_timeseries)
        tau_list.append(tau_i)
        avg_energy=np.mean(energy_timeseries[5*tau_i:])
        average_energy_list.append(avg_energy)


        plt.plot(-np.asarray(average_energy_list))
        plt.plot(np.asarray([exact_data[i][1] for i in range(171)]))
