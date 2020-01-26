#import required libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


#Function to find relation between all data parameters 
def scatter_plot (data):
    scatter_matrix_plot = scatter_matrix(dataset, figsize=(20, 20))
    for ax in scatter_matrix_plot.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize = 7, rotation = 45)
        ax.set_ylabel(ax.get_ylabel(), fontsize = 7, rotation = 90)
    return scatter_matrix_plot


#import test data
dataset = pd.read_csv('Dataset.csv')
dataset4 = pd.read_csv('Dataset.csv')


#Find relation between all data parameters
scatter_matrix_plot = scatter_matrix(dataset, figsize=(20, 20))
for ax in scatter_matrix_plot.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize = 7, rotation = 45)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 7, rotation = 90)
plt.show()


#Using Pearson Correlation find relation between various parameters
plt.figure(figsize=(10,10))
cor = dataset.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


#Find out co-relation between variables
corr_matrix = dataset.corr()
corr_matrix["Dyno_Torque"].sort_values(ascending=False)








#2. Look at linearity and variance with scatterplots
#Scatterplots are an absolutely great tool when it comes to data exploration for regression analysis. Plotting our independent variables against our dependent one on a scatterplot allows us to check for linearity — which is one of the mandatory regression analysis assumptions — , observe the variance on our data — that can cause our model to not hold the homoscedasticity regression assumption, as well as spot outliers. We can also confirm categorical variables by the way they look on a scatterplot as well.
#There are several different ways you can plot a scatterplot, but my favorite is the Seaborn regplot because it plots a linear regression model line on top of our data, making it one step easier to visualize linearity and how outliers might impact our model.




# to make plotting easier we will remove our target variable price from the main dataframe and save it on a separate one:

df = pd.read_csv('Dataset.csv')

target = df.Dyno_Torque
df = df.drop(columns=['Dyno_Torque'])
# we use a for loop to plot our independent variables against our dependent one:
for col in df:
    sns.regplot(x=df[col], y=target, data=df, label=col)
    plt.ylabel('Dyno_Torque')
    plt.xlabel('')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    

# save correlations to variable
corr = dataset.corr()
# we can create a mask to not show duplicate values
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# generate heatmap
plt.figure(figsize= (12,12))
sns.heatmap(corr, annot=True, center=0, mask=mask, cmap='gnuplot')
plt.show()






#Look at data distributions with histograms
#A histogram can give us a quick idea of our data distributions shape, skewness, and scale. It is also useful for starting to have an idea of which are our categorical features. I particularly like to use Seaborn distplot, which adds a kernel density estimate line on top of our histograms. Look at the code and a couple example plots below.
# make sure to import all libraries you'll use for visualizations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# use a for loop to plot each column from your dataframe
for column in dataset:
    sns.distplot(dataset[column], hist_kws=dict(color='plum',    edgecolor="k", linewidth=1))
    plt.show()



# violin plot
for column in dataset:
    sns.violinplot(x=column, data=dataset, color='salmon', whis=3)
    plt.tight_layout()
    plt.show()


#If outliers are an issue, it can also be useful to use a boxplot or a Seaborn violinplot to look at outliers more closely.
# boxplot
plt.figure(figsize=(12,4))
sns.boxplot(dataset['qOil'])
plt.show()








#for cols, title in enumerate(dataset.columns):
 #   print(f" Il y a {classement[title].nunique()} valeurs uniques dans, {title}")
    



# histogramme
 
series = pd.Series()
for cols, title in enumerate(dataset.columns):
    series[title] = dataset[title].nunique()
    
plt.figure(figsize = (10,4), dpi= 100) # Pour gérer la taille de la figure
series.plot(kind = "bar", )











#define X(input) and y(output)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values




#devide data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)





#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #This uses normal equation to calcuate theta for minimum cost function
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#Checking efficiency of model
print('Variance score for training data: %.2f' % regressor.score(X_train, y_train))
print('Variance score for test test: %.2f' % regressor.score(X_test, y_test))



