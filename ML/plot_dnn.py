
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("gonihedric_results_2temps.csv",delimiter="\t",header=None)
df=df.T
#df=df.rename(columns={'0':'Low T','1':'High T','2':'T'},inplace=True)

plt.figure(figsize=(15,10))
ax = plt.gca()
df.plot(kind="line",x=2,y=0,ax=ax,label="Ordered Phase")
df.plot(kind="scatter",x=2,y=1,color='red', ax=ax,label="Disordered Phase")
plt.ylabel('$Probability$')
plt.yticks(np.linspace(0, 1, 15))
plt.xlabel('Temperature')
plt.xticks(np.linspace(1.86, 2.85, 20))
plt.tight_layout()
plt.savefig("critical_temp_dnn_2temps.png",format="png")
plt.show()
plt.close()
