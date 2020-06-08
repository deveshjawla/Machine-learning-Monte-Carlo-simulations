import numpy as np
import random
import time
from numba import jit
import math
import matplotlib.pyplot as plt
import secrets

#h=-J*sum(nnspin interactions)-K*(plaquette interactions)

def single_update(T,L,sweeps):
    one_sweep=L*L
    mag_timeseries=[]
    energy_timeseries=[]
    C_1_timeseries=[]
    C_2_timeseries=[]
    #config_list=[]
    lattice=np.zeros([L,L])
    constant_j=1.0
    constant_k=0.2*constant_j
    for i in range(L):
            for j in range(L):
                lattice[i][j] = secrets.choice([1,-1])

    for sweep in range(sweeps):
        mag=0.0
        C_1=0.0
        C_2=0.0
        C_k=0.0
        energy=0.0

        for mcstep in range(one_sweep):
            del_E_j=0.0
            del_E_k=0.0
            rand=random.SystemRandom().random()
            start_i = secrets.randbelow(L)
            start_j = secrets.randbelow(L)
            flip_spin=lattice[start_i][start_j]
            for (i,j) in [(1,0),(0,1),(-1,0),(0,-1)]:
                del_E_j+= 2.0*constant_j*(flip_spin*lattice[(start_i+i)%L][(start_j+j)%L])
            del_E_k=2.0*flip_spin*constant_k*((lattice[(start_i+1)%L, start_j]*lattice[start_i, (start_j-1)%L]*lattice[(start_i+1)%L, (start_j-1)%L])+(lattice[(start_i-1)%L, start_j]*lattice[start_i, (start_j-1)%L]*lattice[(start_i-1)%L, (start_j-1)%L])+(lattice[(start_i-1)%L, start_j]*lattice[start_i, (start_j+1)%L]*lattice[(start_i-1)%L, (start_j-1)%L])+(lattice[(start_i+1)%L, start_j]*lattice[start_i, (start_j+1)%L]*lattice[(start_i+1)%L, (start_j+1)%L]))
            del_E=del_E_j+del_E_k
            #print(del_E_k,del_E,del_E_j)
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
                C_1+= lattice[i,j]*(lattice[(i+1)%L, j] + lattice[i,(j-1)%L])
                C_2+= lattice[i,j]*(lattice[(i+1)%L, (j+1)%L] + lattice[(i+1)%L, (j-1)%L])
                C_k+= lattice[i,j]*(lattice[(i+1)%L, j]*lattice[i, (j-1)%L]*lattice[(i+1)%L, (j-1)%L])
        C_1/=float(L*L)
        C_2/=float(L*L)
        C_k/=float(L*L)
        energy= -constant_j*C_1-constant_k*C_k

        energy_timeseries.append(energy)
        C_1_timeseries.append(C_1)
        C_2_timeseries.append(C_2)
        #print(lattice)

        #config_list.append(lattice)
        #print(config_list)

    #saving files
    #np.save("metropolis_data/configs/" + str(T) + "_L" + str(L),config_list)

    C_1_timeseries_file = "metropolis_data/" + "C_1_timeseries_" + str(T) + "_L" + str(L) + ".txt"
    np.savetxt(C_1_timeseries_file,C_1_timeseries)
    C_2_timeseries_file = "metropolis_data/" + "C_2_timeseries_" + str(T) + "_L" + str(L) + ".txt"
    np.savetxt(C_2_timeseries_file,C_2_timeseries)
    energy_timeseries_file = "metropolis_data/" + "energy_timeseries_" + str(T) + "_L" + str(L) + ".txt"
    np.savetxt(energy_timeseries_file,energy_timeseries)
#     lattice_config_file = "metropolis_data/" + "config_" + str(T) + "_L" + str(L) + "sweep no." + str(i)
#     np.save(lattice_config_file,lattice)
#     mag_timeseries_file = "metropolis_data/" + "mag_timeseries_" + str(T) + "_L" + str(L) + ".txt"
#     np.savetxt(mag_timeseries_file,mag_timeseries)

    #plotting observables
    plt.plot(energy_timeseries,label="Metropolis_energy_time_series_"+str(T)+"_"+str(L))
#     plt.plot(mag_timeseries,label="Metropolis_Magnetisation_time_series_"+str(T)+"_"+str(L))
#     plt.legend()
    plt.show()
#     plt.close()
    return None

    temp_list=np.linspace(2.493,6,1)
    temp_expo_list=np.exp(-2.0/temp_list)
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
                single_update(T,L,500)
                #cluster_update(T,L,500,expo)
                t_fin = time.time()
                CPU_timematrix[i][j] = t_fin - t_in
                print("done L = " + str(L) + ", T = " + str(T))



        np.savetxt("cputimes3"+ ", T = " +str(T)+"L_"+str(L)+".txt",CPU_timematrix)
    do_and_save()

    import numpy as np
    import random
    import time
    from numba import jit
    import math
    import matplotlib.pyplot as plt
    import secrets

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
    T=2.493
    L=20
    energy_timeseries=np.loadtxt("metropolis_data/" + "energy_timeseries_" + str(T) + "_L" + str(L) + ".txt")
    tau(energy_timeseries)
    # avg_energy_independednt=np.mean(energy_timeseries[5*tau_i::2*tau_i])
    # #np.savetxt("avg_energy_independednt"+ ", T = " +str(T)+"L_"+str(L)+".txt",avg_energy_independednt)
    # avg_energy_dependednt=np.mean(energy_timeseries[5*tau_i:])
    # #np.savetxt("avg_energy_dependednt"+ ", T = " +str(T)+"L_"+str(L)+".txt",avg_energy_dependednt)
    # avg_energy_independednt_eq=np.mean(energy_timeseries[20000::5*tau_i])
    # #np.savetxt("avg_energy_independednt_eq"+ ", T = " +str(T)+"L_"+str(L)+".txt",avg_energy_independednt_eq)
    # print(avg_energy_dependednt,avg_energy_independednt_eq,avg_energy_independednt,tau_i)
    # plt.plot(energy_timeseries[20000::2*tau_i])

    temp_binder_list=[]

    def binderparameter(L,T):
        mag_list=np.loadtxt("metropolis_data/" + "mag_timeseries_" + str(T) + "_L" + str(L) + ".txt")#edit
        maglist=mag_list[5*tau::2*tau]
        maglist_4=maglist**4
        maglist_2=maglist**2
        avg_maglist_4=np.mean(maglist_4[:])#edit
        avg_maglist_2=np.mean(maglist_2[:])#edit
        binder_param=1-(avg_maglist_4/(3*(avg_maglist_2)**2))
        return binder_param

    for i in L_list:
        for j in T_list:
            k=binderparameter(j,i)
            temp_binder_list.append((j,k))
