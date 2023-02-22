import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier

lr=LogisticRegression()
svc=LinearSVC(C=1.0)
rfc=RandomForestClassifier(n_estimators=100)

df=pd.read_csv("Dataset.csv")

from sklearn.model_selection import train_test_split

train,test= train_test_split(df,test_size=0.3)

train_feat=train.iloc[:,:2]
train_targ=train["Pred"]

print("{0:0.2f}% in training set".format((len(train_feat)/len(df.index)) * 100))
print("{0:0.2f}% in testing set".format((1-len(train_feat)/len(df.index)) * 100))

seed = 7
num_trees = 10
kfold = model_selection.KFold(n_splits=100, random_state=None)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model,train_feat,train_targ, cv=kfold)
acc=results.mean()
acc1=acc*100
print("The accuracy is: ",acc1,'%')

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(train_feat, train_targ)
y_2 = model.predict(train_feat)
#df['y_predicted'] = y_2

# Calculating the MSE with sklearn
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(train_targ, y_2)
print(mse)

from sklearn.metrics import r2_score  
R_square = r2_score(train_targ, y_2) 
print('Coefficient of Determination', R_square)

from sklearn.metrics import mean_squared_error
rms = mean_squared_error(train_targ, y_2, squared=False)
print(rms)
