```python
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import tree, metrics

data = pd.read_csv("Train_v2.csv")

data.info()
```


```python
data = data.drop(['uniqueid'], axis=1)
```


```python
data.head()
```


```python
data.shape
```


```python
class_names = data['bank_account']
data['bank_account'], _ = pd.factorize(data['bank_account'])


```


```python
data['bank_account'], _ = pd.factorize(data['bank_account'])
data['country'],_ = pd.factorize(data['country'])
data['year'],_ = pd.factorize(data['year'])
data['location_type'],_ = pd.factorize(data['location_type'])
data['cellphone_access'],_ = pd.factorize(data['cellphone_access'])
data['household_size'],_ = pd.factorize(data['household_size'])
data['age_of_respondent'],_ = pd.factorize(data['age_of_respondent'])
data['gender_of_respondent'],_ = pd.factorize(data['gender_of_respondent'])
data['relationship_with_head'],_ = pd.factorize(data['relationship_with_head'])
data['marital_status'],_ = pd.factorize(data['marital_status'])
data['education_level'],_ = pd.factorize(data['education_level'])
data['job_type'],_ = pd.factorize(data['job_type'])
data.head()
```


```python
data.info()
```


```python

data.shape
```


```python
x = data.loc[:, data.columns != 'bank_account']
y = data['bank_account']
y

```


```python
x.info()

```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)
```


```python
dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=0)
dtree.fit(X_train, y_train)

#use the model to make prediction
y_pred = dtree.predict(X_test)
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
```


```python
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
```


```python
import graphviz
feature_names = x.columns
dot_data = tree.export_graphviz(dtree, out_file=None, filled=True, rounded=True,
feature_names=feature_names,
class_names=class_names)
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/icom/.conda/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz'
graph = graphviz.Source(dot_data)
graph
```
