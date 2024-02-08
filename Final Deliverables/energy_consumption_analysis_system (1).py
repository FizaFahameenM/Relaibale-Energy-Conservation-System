# -*- coding: utf-8 -*-
"""Energy Consumption Analysis System.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10scNo9_2rpnFzbvRSpMx8fGF5E1uRfDX

# A Reliable Energy Consumption Analysis System for Energy-Efficient Appliances

## Importing Packages
"""

import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn import metrics 
from sklearn.model_selection import cross_val_score
import pickle
from flask import Flask,request,render_template

"""## importing Dataset"""

dt = pd.read_csv('/content/drive/MyDrive/household_power_consumption.txt',sep = ';',
                parse_dates={'dt':['Date','Time']},
                infer_datetime_format=True,
                low_memory=False, na_values=['nan','?'],
                index_col='dt')

dt.info()

dt.isnull().sum()

dt.replace('?',np.nan,inplace=True)

dt.loc[dt.Sub_metering_3.isnull()].head

dt = dt.dropna(how = 'all')

for i in dt.columns:
  dt[i] = dt[i].astype("float64")

values = dt.values
dt['Sub_metering_4'] = (values[:,0]*1000/60)-(values[:,4]+ values[:,5]+values[:,6])
dt.shape

dt.dtypes

dt.corr()

dt.describe()

"""## Uni-Variate"""

sns.displot(dt['Global_active_power'])

sns.distplot(dt['Global_reactive_power'],kde=False,bins=30)

sns.distplot(dt['Global_active_power'],kde=False,bins=30)

"""## Bivariate Analysis"""

sns.jointplot(x = 'Global_reactive_power',y = 'Global_active_power',data = dt,kind = 'scatter')

sns.jointplot(x = 'Voltage',y = 'Global_active_power',data = dt,kind = 'scatter')

sns.jointplot(x = 'Global_intensity',y = 'Global_active_power',data = dt,kind = 'scatter')

sns.jointplot(x = 'Sub_metering_1',y = 'Global_active_power',data = dt,kind = 'scatter')

sns.jointplot(x = 'Sub_metering_3',y = 'Global_active_power',data = dt,kind = 'scatter')

sns.jointplot(x = 'Sub_metering_4',y = 'Global_active_power',data = dt,kind = 'scatter')

"""## Multivariate analysis"""

pearson = dt.corr(method='pearson')
mask = np.zeros_like(pearson)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(pearson, vmax=1,vmin=0,square=True,cbar=True,annot=True,cmap="YlGnBu",mask=mask)

X = dt.iloc[:,[1,3,4,5,6]]
y = dt.iloc[:,0]
X.head()

y.head()

"""## Splitting train and test Dataset"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

"""## Training The Model In Multiple Algorithms

### Linear Regression model
"""

lm=LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)

predictions

"""### XGB Regressor


"""

import xgboost as xgb
model2 = xgb.XGBRegressor()
model2.fit(X_train,y_train)
y_predict2 = model2.predict(X_test)

y_predict2

"""### Random Forest Regressor Model"""

model3 = RandomForestRegressor()
model3.fit(X_train,y_train)
y_predict3 = model3.predict(X_test)

y_predict3

"""### Ridge Regressor Model


"""

model4 = Ridge()
model4.fit(X_train, y_train)
y_pred_ridge = model4.predict(X_test)

y_pred_ridge

"""## Prediction

LINEAR
"""

y_p1 = lm.predict([[0.148,18.4,0.0,1.0,17.0]])
y_p1

"""Random Forest Regressor Model"""

y_p3 = model3.predict([[0.148,18.4,0.0,1.0,17.0]])
y_p3

y_p4 = model4.predict([[0.148,18.4,0.0,1.0,17.0]])
y_p4

"""## Testing Model With Multiple Evaluation Metrics

LINEAR
"""

print('MAE:' ,metrics.mean_absolute_error(y_test,predictions))
print('MSE:',metrics.mean_squared_error(y_test,predictions))
print('RMSE:' ,np.sqrt(metrics.mean_squared_error(y_test,predictions)))
print('RSquarevalue:' ,metrics.r2_score (y_test,predictions))

"""XGB Regressor

"""

print('MAE:' ,metrics.mean_absolute_error(y_test,y_predict2))
print('MSE:',metrics.mean_squared_error(y_test,y_predict2))
print('RMSE:' ,np.sqrt(metrics.mean_squared_error(y_test,y_predict2)))
print('RSquarevalue:' ,metrics.r2_score (y_test,y_predict2))

"""Random Forest Regressor Model"""

print('MAE:' ,metrics.mean_absolute_error(y_test,y_predict3))
print('MSE:',metrics.mean_squared_error(y_test,y_predict3))
print('RMSE:' ,np.sqrt(metrics.mean_squared_error(y_test,y_predict3)))
print('RSquarevalue:' ,metrics.r2_score (y_test,y_predict3))

"""Ridge model"""

print('MAE:' ,metrics.mean_absolute_error(y_test,y_pred_ridge))
print('MSE:',metrics.mean_squared_error(y_test,y_pred_ridge))
print('RMSE:' ,np.sqrt(metrics.mean_squared_error(y_test,y_pred_ridge)))
print('RSquarevalue:' ,metrics.r2_score (y_test,y_pred_ridge))

"""## Comparing Model Accuracy Before & After Applying Hyperparameter Tuning"""

cv = cross_val_score(lm,X,y,cv=5)

np.mean(cv)

"""## Save The Best Model

"""

filename = 'PCASSS_model.pkl'
pickle.dump(lm,open(filename,'wb'))

"""## Integrate With Web Framework

"""

model = pickle.load(open('PCASSS_model.pkl','rb'))
app = Flask(__name__)

@app.route("/")
def f():
  return render_template("index.html")

@app.route("/inspect")
def inspect():
  return render_template("inspect.html")

@app.route("/home", methods=["GET", "POST" ])
def home():
  GlobalReactivePower = float(request.form['GlobalReactivePower'])
  Global_intensity - float(request.form['Global_intensity'])
  Sub_metering_1 - float(request.form['Sub_metering_1'])
  Sub_metering_2 - float(request.form['Sub_metering_2'])
  Sub_metering_3 = float(request.form['Sub_metering_3'])
  X = [[GlobalReactivePower,Global_intensity,Sub_metering_1,Sub_metering_2,Sub_metering_3]]
  output = round(model.predict (X) [0], 3)
  return render_template('output .htmI',output=output)

if __name__ == "__main__":
  app.run(debug=True)