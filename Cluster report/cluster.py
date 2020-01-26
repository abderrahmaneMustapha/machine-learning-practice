import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('student_master.csv')
branch = pd.read_csv('student_master_branchs.csv')
data = data.drop(columns=['id_student'])
branch = branch.drop(columns=['id_student'])

x = data.copy()
y = branch.copy()


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data_len = x.shape[1]
for i in range(data_len):
    x[:,i] = label.fit_transform(x[:,i])
y = label.fit_transform(y)
