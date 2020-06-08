import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import timeit
plt.figure(figsize=(15,15))
#config_list=list_configs[20000::16]
T=5.0
L=20

Y=np.asarray(np.loadtxt("metropolis_data/" + "energy_timeseries_" + str(T) + "_L" + str(L) + ".txt")[2000::16])
X=np.asarray([np.loadtxt("metropolis_data/" + "C_1_timeseries_" + str(T) + "_L" + str(L) + ".txt")[2000::16],np.loadtxt("metropolis_data/" + "C_2_timeseries_" + str(T) + "_L" + str(L) + ".txt")[2000::16]])
#Y is the Energy timeseries
#X=np.array([123,4324,234],[345,345345,34345]) is 2d array with C_1 and C_2 interaction timeseries
def fit_func(X,a,b,c):#a=E_0,b=J_1,c=J_2
    x,y=X
    return a-(b*x)-(c*y)

def chi2(ydata,Nparams,f,*params):#Reduced Chi-Square without y-error
    res = ydata - f(*params)
    redchi2 = sum(res**2)/(len(ydata)-Nparams)
    return res, redchi2

def chi2abs(ydata,yErr,Nparams,f,*params):#Reduced Chi-Square with y-error
    res = ydata - f(*params)
    redchi2 = sum((res/yErr)**2)/(len(ydata)-Nparams)
    return res, redchi2

popt_1,pcov_1=curve_fit(fit_func,X,Y)

r,redchi2 = chi2(Y,3,fit_func,X,*popt_1)
print('Parameters J_{1} and J_{2}\n',popt_1,'\n Covariance\n',pcov_1,'\n Chi-Square=\n',redchi2,'\n\n')
plt.scatter(X[0],fit_func(X,*popt_1),label='Fit',color='r',s=1)

plt.scatter(X[0],Y,label='Data',color='g',marker='.',s=1)
plt.xlabel('C_1/N')
plt.ylabel('Energy/N')
plt.legend()
#plt.savefig('Graph of fitting with all parameters.png')
plt.show()
