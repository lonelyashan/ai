import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

ds=pd.read_csv("study_performance.csv")
dsp= pd.get_dummies(ds, drop_first=True)

x = dsp.drop('math_score', axis=1)
y = dsp['math_score']

x0, x1, y0, y1 = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"размер обучающей выборки: {x0.shape}")
print(f"размер тестовой выборки: {x1.shape}")

model = LinearRegression()
model.fit(x0, y0)

ypred = model.predict(x1)
mse = mean_squared_error(y1, ypred)
r2 = r2_score(y1, ypred)

print(f"mean_squared_error: {mse:.2f}")
print(f"r2_score: {r2:.2f}")

ds_class = ds.copy()
ds_class['math_passed'] = (ds_class['math_score'] >= 60).astype(int)

xc = pd.get_dummies(ds_class.drop(['math_score', 'math_passed'], axis=1), drop_first=True)
yc = ds_class['math_passed']

x0c, x1c, y0c, y1c = train_test_split(xc, yc, test_size=0.2, random_state=42)

modelc = LogisticRegression(max_iter=1000)
modelc.fit(x0c, y0c)
ypredc = modelc.predict(x1c)

acc = accuracy_score(y1c, ypredc)
print(f"accuracy math_score: {acc:.2f}")
