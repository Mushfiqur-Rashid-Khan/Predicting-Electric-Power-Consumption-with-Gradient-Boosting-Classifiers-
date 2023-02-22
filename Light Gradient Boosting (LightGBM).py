import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization

# load and preview data
df = pd.read_csv('Dataset.csv')
df.head()

df['Pred'].value_counts()

X = df[['power']]
y = df['Pred']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# build the lightgbm model
import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)

# predict the results
y_pred=clf.predict(X_test)

# view accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

# Calculating the MSE with sklearn
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(mse)

from sklearn.metrics import r2_score  
R_square = r2_score(y_test, y_pred) 
print('Coefficient of Determination', R_square)

from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_test, clf.predict(X_test))
print(mape)

from sklearn.metrics import mean_squared_error
rms = mean_squared_error(y_test, y_pred, squared=False)
print(rms)
