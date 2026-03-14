import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

ds=pd.read_csv("study_performance.csv")
ds.loc[0:2, 'reading_score'] = np.nan
ds.loc[10:15, 'math_score'] = np.nan

print(ds.head())
print("\n")

print(ds.isnull().sum())
print("\n")
ds['reading_score'] = ds['reading_score'].fillna(ds['reading_score'].median())

ds['math_score'] = ds['math_score'].fillna(ds['math_score'].mode()[0])

print(ds.isnull().sum())
print("\n")
scaler = MinMaxScaler()
cols = ['math_score', 'reading_score', 'writing_score']
ds[cols] = scaler.fit_transform(ds[cols])

print(ds.head())
print("\n")

ds = pd.get_dummies(ds, columns=['gender'], drop_first=True)
print(ds.head())
