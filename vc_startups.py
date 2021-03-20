# creating model for the VC_Startup dataset

# importing important libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# get current working directory
os.getcwd()
# 'C:\\Users\\Shubham'

# changing the current working directory
os.chdir('C:\\Users\\Shubham\\Desktop\\Python\\VC_Startups')

# reading dataset
dataset = pd.read_csv('VC_Startups.csv')
dataset

# after reading the dataset our aim should be to understant the data
# profit is dependent feature and rest all are independent
# for understanding the data we use seaborn library
# importing seaborn lib

import seaborn as sns

# now taking individual features for the visulization and understanding purpose

rnd = dataset.iloc[:, 0]
rnd

type(rnd) # Series

x = rnd.values
x
profit = dataset.iloc[:,-1] # dependent feature
profit

y = profit.values
y

# using scatter plot 
plt.scatter(x,y,color = 'k', s = 100) # s --> used --> increase the density of color
# x label
plt.xlabel('R&D spend')
# y label
plt.ylabel('Profit')
# title
plt.title('Profit VS R&D spend')
plt.show()

# from plot we see that there is positive corelation between profit and r&d spend

# now we will do the analysis for remaining of the features 

dataset.columns # for checking the features

admin = dataset.iloc[:,1]
x = admin.values

# again using scatter plot
plt.scatter(x,y,color='b')
plt.xlabel('Administration') # x label
plt.ylabel('Profit') # y label
plt.title('Administration VS Profit') # title

# from plot we see that there is no relation between them as points are scattered improperly
# so we can remove the admistration feature

mrkt_spend = dataset.iloc[:,2]
x = mrkt_spend.values

plt.scatter(x,y,color = 'r',s = 100)
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.title('Marketing Spend VS Profit')

# from plot we see that there is positive corelation between them

# now for last feature 
state = dataset.iloc[:,-2]
x = state.values

plt.scatter(x,y,color = 'b', s = 100)
plt.xlabel('State')
plt.ylabel('Profit')
plt.title('State VS Profit')

# for checking the count of the State
sns.countplot('State',data=dataset)
# # now checking the rnd and profit with respect to state
# sns.barplot('Profit',hue='State',data = dataset)

df = dataset.iloc[:,3:5]
df.boxplot(column = 'Profit', by = 'State')


# from this plot we see that state have impact on the profit

# So, 
        # R&D Spend, Marketing Spend, State ---> important
        # Administration ---> not important

# lets see by plot that are these features related to each other or not
plt.scatter(rnd.values,mrkt_spend)

# from plot --> they are related with each other (positively)




# check for null or missing values in the dataset
missing_values = dataset.isnull().sum()
missing_values[missing_values>0]

# there is no null or any missing value in the dataset

# Now creating model for our dataset

# sklearn 

###

# taking independent features
X = dataset.iloc[:,:-1]
X

# separating or taking dependent feature
y = dataset.iloc[:,-1]
y

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)

# Linear Regressor
# import LinearRegression from sklearn.linear_model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# training our data

regressor.fit(X_train,y_train)
'''
There will an error while training the data because the dtype
of one column is string and our machine learning model does not 
accept the string type i.e., it only accepts the numerical values.

So, either we can drop that column or we use Encoding techniques for 
converting string to numeric values.

    Encoding Techniques----> converts ---> str --> to --> numeric values
'''
# predicting

regressor.predict(X_test)


dataset.State


X_train

###

# firstly, removing or dropping the column with object dtype

X = dataset.iloc[:,:-2]
X

y = dataset.iloc[:,-1]
y

# partitioning the data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.2, random_state = 0)

# importing LinearRegression from sklearn.linear_model
from sklearn.linear_model import LinearRegression
# creating obj
regressor = LinearRegression()

# train 
regressor.fit(X_train,y_train)

#predict
predicted = regressor.predict(X_test)

# checking the accuracy of the model
# for this import accuracy_score from sklearn.metrics

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test.values,predicted)
print(score)


y_test.values


s = (np.median(y_test)-np.median(predicted))/np.median(y_test)
s
# to overcome this error or preventing from this error we use encoding techniques

# Since our data is categorical , so we can use OneHotEncoder

# Dummy vars and encoders

# for this import OneHotEncoder and LabelEncoder from sklearn.preprocessing

from numpy import array
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# first of all lets check what actually happens with all these encoding techniques

# creating own dataset2

dataset2 = ['Pizza','Burger','Bread','Bread','Bread','Burger','Pizza','Burger']
dataset2

values = array(dataset2)
print(values)

# creating obj for label encoder
label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(values)
# this assigns the rank to the values as per the order
integer_encoded
# we can see that the string is converted to numerical values

integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
print(integer_encoded)

# now creating obj for one hot encoder
onehot = OneHotEncoder(sparse = False)

onehot_encoded = onehot.fit_transform(integer_encoded)

print(onehot_encoded)
# we can see that the output is differentiated wrt to categorical data

# now working with our dataset

X = dataset.iloc[:,:-1].values
X

y = dataset.iloc[:,-1].values
y

# so now before partitioning we must encode our dataset

# Encoding the categorical data

label_encoder = LabelEncoder()
X
X[:,3] = label_encoder.fit_transform(X[:,3])
X

# we see that str converted to numercial data as per order 

# one hot encoder
X
onehot = OneHotEncoder(categorical_features = [3])

tmpDF = pd.DataFrame(X)
tmpDF
X
X = onehot.fit_transform(X).toarray()
X

tmpDF = pd.DataFrame(X)
tmpDF

X
X.shape

X[:,1:].shape
X.shape

# Avoiding Dummy Variable Trap
# n-1 

X = X[:,1:]
X.shape

# splitting the data into training and test data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size= 0.2, random_state = 0)

# creating obj for model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
# train 
regressor.fit(X_train,y_train)
#predict 
predicted = regressor.predict(X_test)
predicted


# these are for classification problem statements
###
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, predicted)
print(score)


from sklearn.metrics import confusion_matrix
score = confusion_matrix(y_test, predicted)
print(score)
###
y_test
predicted
s = (np.median(y_test)-np.median(predicted))/np.median(y_test)
s






