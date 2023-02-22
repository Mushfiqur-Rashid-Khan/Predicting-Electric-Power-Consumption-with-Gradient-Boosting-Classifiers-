import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
dataset=pd.read_csv('Dataset.csv')
from sklearn.model_selection import train_test_split
X=dataset.iloc[:,:2]
y=dataset["Pred"]

seed = 5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

model = XGBClassifier()
model.fit(X_train, y_train)
print(model)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Calculating the MSE with sklearn
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(mse)

from sklearn.metrics import r2_score  
R_square = r2_score(y_test, y_pred) 
print('Coefficient of Determination', R_square)

from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test, model.predict(X_test))

print(mape)

from sklearn.metrics import mean_squared_error

rms = mean_squared_error(y_test, y_pred, squared=False)
