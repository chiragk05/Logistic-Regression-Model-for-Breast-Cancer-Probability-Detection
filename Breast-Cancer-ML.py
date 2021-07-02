import seaborn as sns 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn 
from sklearn import *

#Read the .csv file, in my case it is named as; 'data.csv'
data = pd.read_csv('data.csv', index_col=False)

#A preview into what our data looks like:
data.head(5)

#Get a look into the shape of your data: 
data.shape

#Remove the un-needed data: 
data = data.drop('id', axis=1)
data = data.dropna(axis=1)

#Plot the Malignant ('M') and Benign ('B') cases
sns.countplot(data['diagnosis'], label='count', palette=['#1ABC9C','#3498DB'])

#Get a look into the data types in our data
data.dtypes

#View the diagnoses as 'M' or 'B'
data['diagnosis'].values

#Convert them into the binary: 'M' as 1 and 'B' as 0
data.loc[(data.diagnosis == 'M'),'diagnosis']= int(1)
data.loc[(data.diagnosis == 'B'),'diagnosis']= int(0)
data['diagnosis'].values

#Try plotting the 'texture_mean' against the 'diagnosis'
sns.relplot(x='texture_mean', y='radius_mean', hue='diagnosis', data=data)

#Get the correlation values between the different attributes
data.corr()

#Draw up the heat-map
plt.figure(figsize=(20,20))  
sns.heatmap(data.corr(), annot=True, fmt='.0%')

#The first step into the splitting up the data into X and Y; independent and dependent 
x = data.iloc[:,1:31].values
y = data.iloc[:,0].values

#Change the Y - Values to integers 
y = y.astype('int')

#Split the model into 'testing' and 'training' data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#Scale the dependent attributes to the same magnitude:
from sklearn.preprocessing import StandardScaler
StdS = StandardScaler()
X_train = StdS.fit_transform(X_train)
X_test = StdS.fit_transform(X_test)

#Create your Logistic Model:
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, Y_train)

#Get the accuracy of your model: 
acc = logreg.score(X_test, Y_test)
print(acc)

#Save you model: 
import pickle
with open("breastcancermodel.pickle", "wb") as f:
  pickle.dump(logreg, f)

#Load in your new data to test-out your model: 
new_test_data = pd.read_csv('new_x_test.csv')
predicted = model.predict(new_test_data)

#Print your predicted outcome and compare it with the actual outcome: 
print(predicted)
