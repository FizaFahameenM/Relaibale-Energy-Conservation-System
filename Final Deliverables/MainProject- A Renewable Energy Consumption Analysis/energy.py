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


##importing data 

dt = pd.read_csv(r'C:\Users\HP\Downloads\household_power_consumption.txt.zip',sep = ';',
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

X = dt.iloc[:,[1,3,4,5,6]]
y = dt.iloc[:,0]
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

lm=LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
predictions

y_p1 = lm.predict([[0.148,18.4,0.0,1.0,17.0]])
y_p1

print('MAE:' ,metrics.mean_absolute_error(y_test,predictions))
print('MSE:',metrics.mean_squared_error(y_test,predictions))
print('RMSE:' ,np.sqrt(metrics.mean_squared_error(y_test,predictions)))
print('RSquarevalue:' ,metrics.r2_score (y_test,predictions))

cv = cross_val_score(lm,X,y,cv=5)
np.mean(cv)

##FLASK PART
filename = 'PCASSS_model.pkl'
pickle.dump(lm,open(filename,'wb'))

model = pickle.load(open('PCASSS_model.pkl','rb'))
app = Flask(__name__)

@app.route("/")
def f():
  return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')


@app.route("/inspect")
def inspect():
  return render_template("inspect.html")

@app.route("/inspect", methods=["GET", "POST"])
def inspect_data():
    if request.method == "POST":
        GlobalReactivePower = float(request.form['input1'])
        Global_intensity = float(request.form['input2'])
        Sub_metering_1 = float(request.form['input3'])
        Sub_metering_2 = float(request.form['input4'])
        Sub_metering_3 = float(request.form['input5'])
        X = [[GlobalReactivePower, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3]]
        output = round(model.predict(X)[0], 3)
        return render_template('output.html', output=output)
    else:
        return render_template('inspect.html')


# @app.route("/home", methods=["GET", "POST" ])
# def home():
#   GlobalReactivePower = float(request.form['GlobalReactivePower'])
#   Global_intensity = float(request.form['Global_intensity'])
#   Sub_metering_1 = float(request.form['Sub_metering_1'])
#   Sub_metering_2 = float(request.form['Sub_metering_2'])
#   Sub_metering_3 = float(request.form['Sub_metering_3'])
#   X = [[GlobalReactivePower,Global_intensity,Sub_metering_1,Sub_metering_2,Sub_metering_3]]
#   output = round(model.predict (X) [0], 3)
#   return render_template('output.html',output=output)

if __name__ == "__main__":
  app.run(debug=True ,host='0.0.0.0',port = 8000)