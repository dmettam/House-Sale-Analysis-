# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 22:55:16 2020

@author: Dane Mettam
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
%matplotlib inline

file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)

# used in console to get an idea of what data is 
# also making sure types make sence
# there are 21,613 rows and 22 columns 
df.head()
df.shape 
df.dtypes 
df.describe() 


# dropping id and unnamed: 0 since that data is not useful 
df.drop(['id'], axis=1, inplace=True)
df.drop(['Unnamed: 0'], axis=1, inplace=True) 
df.describe()

# date is type object, changing it to datetime64 format 
df["date"] = pd.to_datetime(df["date"])
df.head() # to check change 

# checking for missing values in data 
list(df) #viewed first to see all column names for reference 
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum()) 
print("number of NaN values for the column sqft_living :", df['sqft_living'].isnull().sum())
print("number of NaN values for the column sqft_lot :", df['sqft_lot'].isnull().sum()) 
print("number of NaN values for the column floors :", df['floors'].isnull().sum())
print("number of NaN values for the column waterfront :", df['waterfront'].isnull().sum()) 
print("number of NaN values for the column view :", df['view'].isnull().sum())
print("number of NaN values for the column condition :", df['condition'].isnull().sum()) 
print("number of NaN values for the column grade :", df['grade'].isnull().sum())
print("number of NaN values for the column sqft_above :", df['sqft_above'].isnull().sum()) 
print("number of NaN values for the column sqft_basement :", df['sqft_basement'].isnull().sum())
print("number of NaN values for the column yr_built :", df['yr_built'].isnull().sum()) 
print("number of NaN values for the column yr_renovated :", df['yr_renovated'].isnull().sum())
print("number of NaN values for the column zipcode :", df['zipcode'].isnull().sum()) 
print("number of NaN values for the column bedrooms :", df['lat'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['long'].isnull().sum()) 
print("number of NaN values for the column bedrooms :", df['sqft_living15'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['sqft_lot15'].isnull().sum()) 

# removing rows with null data in bedrooms and bathrooms 
df['bedrooms'].dropna(axis=0, inplace=True)
df['bathrooms'].dropna(axis=0, inplace=True) 

# check to make sure they dropped 
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum()) 

# to give a rough idea of how many floor options there are
floor_counts = df["floors"].value_counts().to_frame()
floor_counts

# multiple box plots and a violin plot shows that number of floors is
# not a good indicator of price but whether or not it is waterfront 
# property or not is a good indicator that shifts the price up. 
floor_vio1 = sns.violinplot(x="floors", y="price", data=df)
floor_vio2 = sns.violinplot(x="floors", y="price", hue="waterfront", data=df)


# number of bathrooms is also a great indicator but there is clearly 
# another value that is helping predict values
bathbox1 = sns.boxplot(x="bathrooms", y="price", data=df)
bathbox2 = sns.boxplot(x="bathrooms", y="price", hue="waterfront", data=df)

# To see which aspects impact price most 
# I was surprised to see that zipcode was not a good indicator 
# I plan on changing zipcode to string variable and trying to 
# plot it in different ways. 
df.corr()['price'].sort_values()

# to see the aspects with the greatest impact on price 
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms",
           "sqft_living15","sqft_above","grade","sqft_living"]   
X = df[features]
Y = df['price']

lm = LinearRegression()
# The code (lm.fit(X,Y)) worked online but isn't working on Spyder (Python 3.8)
# Debugging this now. 
# It says 'ValueError: Input contains NaN, infinity or 
# a value too large for dtype('float64')
# If someone on GitHub see's this and knows what the issue is, please tell me. 
# Thank you 
lm.fit(X,Y)
lm.score(X, Y)





