import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('3.12. Example.csv')
"""
plt.scatter(data['Satisfaction'], data['Loyalty'])
plt.xlabel('Satisfation')
plt.ylabel('loyalty')
plt.show()
"""

x = data.copy()

kmean = KMeans(2)
kmean.fit(x)

clusters = x.copy()
clusters['cluster_pred']=kmean.fit_predict(x)

"""
plt.scatter(clusters['Satisfaction'], clusters['Loyalty'],c=clusters['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfation')
plt.ylabel('loyalty')
plt.show()
"""
from sklearn.metrics import accuracy_score

from sklearn import preprocessing
x_scaled=preprocessing.scale(x)

wss=[]

for i in range(1,30):
    kmean=KMeans(i)
    kmean.fit(x_scaled)
    wss.append(kmean.inertia_)

print(wss)

#visualise elbow
""""
plt.plot(range(1,30),wss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
"""

