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



