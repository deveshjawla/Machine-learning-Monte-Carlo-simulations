import pandas as pd
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

df=pd.read_csv("correlation_data.csv")
features=df.drop(columns=["kappa","Temperature"])
columns_names=features.columns.tolist()

minmaxscalar=preprocessing.MinMaxScaler()
scaled_features=minmaxscalar.fit_transform(features)
df_scaled_features=pd.DataFrame(scaled_features, columns=columns_names)

standardizer=preprocessing.StandardScaler()
standardized_features=standardizer.fit_transform(features)
df_standardized_features=pd.DataFrame(standardized_features, columns=columns_names)

# correlation_features=df_standardized_features.corr()

# mask = np.tril(np.ones_like(correlation_features, dtype=np.bool))
# plt.figure(figsize=(20,18))
# cmap = sns.diverging_palette(110, 230, as_cmap=True, s=500 ,l=50, n=501, center="light")
# sns.heatmap(correlation_features,cmap=cmap,mask=mask, vmax=1, center=0, square=True, linewidths=0, annot=False)
# plt.title('Correlation between different fearures')
# plt.savefig("Correlation_features",format="png",)
# plt.show()
# plt.close()

pca=PCA()
pca.fit_transform(standardized_features)
covariance=pca.get_covariance()
explained_variance=pca.explained_variance_ratio_
with plt.style.context('dark_background'):
plt.figure(figsize=(12, 4))
plt.bar(range(89), explained_variance, alpha=0.5, align='center', label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("Explained_variance.png",format="png")
plt.show()
plt.close()


pca=PCA(n_components=4)
features_new=pca.fit_transform(standardized_features)
# features_new
pca.get_covariance()
explained_variance=pca.explained_variance_ratio_
# explained_variance
with plt.style.context('dark_background'):
    plt.figure(figsize=(12, 8))
    plt.bar(range(4), explained_variance, alpha=0.5, align='center',label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("Explained_variance_4_comps.png",format="png")
    plt.show()
    plt.close()


from sklearn.cluster import KMeans
clf = KMeans(n_clusters=2)
X=features_new
clf.fit(X)
kappa=df["kappa"].to_numpy()
T=df["Temperature"].to_numpy()
# correct = 0
# for i in range(len(X)):
#     predict_me = np.array(X[i].astype(float))
#     predict_me = predict_me.reshape(-1, len(predict_me))
#     prediction = clf.predict(predict_me)
#     if prediction[0] == y[i]:
#         correct += 1

# print(correct/len(X))
centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.","r.","c.","y."]

plt.figure(figsize=(15,10))
for i in range(len(X)):
    #     print("coordinate:",X[i], "label:", labels[i])
    #     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 1)
    plt.plot(T[i], kappa[i], colors[labels[i]], markersize = 15)
    plt.ylabel('$\kappa$')
    plt.yticks(np.linspace(0, 1, 15))
    plt.xlabel('Temperature')
    plt.xticks(np.linspace(1.72, 3.21, 16))
    plt.tight_layout()
    plt.savefig("kmeans_goniheric.png",format="png")
    plt.show()
    plt.close()

    # plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)


from sklearn.cluster import Birch
clf = Birch()
X=features_new
clf.fit(X)
centroids = clf.subcluster_centers_
labels = clf.labels_
np.unique(labels)
colors = ["g.","r.","c.","y."]
plt.figure(figsize=(20,18))
for i in range(len(X)):
    #     print("coordinate:",X[i], "label:", labels[i])
    #     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 1)
    plt.plot(T[i], kappa[i], colors[labels[i]])


    # plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

    plt.show()
    plt.close()

from sklearn.cluster import MeanShift
clf = MeanShift(n_jobs=4)
# X=features.to_numpy()
X=features_new
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_
np.unique(labels)
colors = ["g.","r.","c.","y.",'k.','y.','m.']
plt.figure(figsize=(15,10))
for i in range(len(X)):
    #     print("coordinate:",X[i], "label:", labels[i])
    #     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
    plt.plot(T[i], kappa[i], colors[labels[i]],markersize = 15)


    # plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
    plt.ylabel('$\kappa$')
    plt.yticks(np.linspace(0, 1, 15))
    plt.xlabel('Temperature')
    plt.xticks(np.linspace(1.72, 3.21, 16))
    plt.tight_layout()
    plt.savefig("meanshift_goniheric.png",format="png")
    plt.show()
    plt.close()


tau_list=np.loadtxt("tau_list.txt")
#tau_list=50000*np.ceil(tau_list/50000)
tau_list.max()
min_max_scaler = preprocessing.StandardScaler()
tau_scaled = min_max_scaler.fit_transform(tau_list.reshape(-1, 1))
tau_scaled.reshape(1,-1)

data=pd.DataFrame({"temperature":T,"kappa":kappa,"z":tau_list.reshape(1,-1)[0]})

oo=data.pivot_table(index='kappa', columns='temperature', values='z')
ooo=oo.reindex(index=oo.index[::-1])
plt.figure(figsize=(20,18))
# cmap = sns.diverging_palette(110, 230, s=500 ,l=50, n=501, center="light")
sns.heatmap(ooo)
#plt.savefig("tau_heatmap.png",format="png")
plt.show()
plt.close()
