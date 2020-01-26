import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import tree, metrics

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
data = pd.read_csv("car.csv", names=['buying', 'maint', 'doors','persons','lug_boot','safety', 'class' ])

print('\n data info')
#get more info about data
print(data.info())

#identify the target variable
data['class'], class_names = pd.factorize(data['class'])

#check encoded values
print(class_names)
print(data['class'].unique())

#identify the predicator
data['buying'],_ =pd.factorize(data['buying']) 
data['maint'],_ =pd.factorize(data['maint']) 
data['doors'],_ =pd.factorize(data['doors']) 
data['persons'],_ =pd.factorize(data['persons']) 
data['lug_boot'],_ =pd.factorize(data['lug_boot']) 
data['safety'],_ =pd.factorize(data['safety']) 
print(data.head())

#check data type after factorize
print(data.info())

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

#split data randomly into 70% trainnig 30% testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)


#train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(X_train, y_train)

#use the model to make prediction
y_pred = dtree.predict(X_test)

#how did our model perform

count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

import graphviz
feature_names = X.columns
dot_data = tree.export_graphviz(dtree, out_file=None, filled=True, rounded=True,
feature_names=feature_names,
class_names=class_names)
graph = graphviz.Source(dot_data)
print(graph.source)
graph.render("Source.gv", view=True)