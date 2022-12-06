from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
import geopandas as gpd 
from sklearn.preprocessing import StandardScaler
import math
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import seaborn as sns
import matplotlib.pyplot as plt
# from werkzeug.utils import secure_filename
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.metrics import mean_squared_error
# from flask import Flask,redirect,render_template,url_for,request
# import numpy as np

# Reading the dataset

earthquake_data = pd.read_csv('earthquake_dataset.csv')
print(earthquake_data)

#Check for the columns with null values
print("The fields with null values are \n",earthquake_data.isnull().sum())

#Check the columns and fill up the null values with 0
earthquake_data = earthquake_data.fillna(0)
print(earthquake_data.isnull().sum())
# earthquake_data

#Trimming only city of the place where earthquake got hit
def place_triming_from_dataset(place):
    if len(place.split(', '))== 1:
        return place.split(', ')[0]
    else:
        return place.split(', ')[1]

earthquake_data["place"] =  earthquake_data["place"].apply(place_triming_from_dataset)
print(earthquake_data["place"])

#Trimming only date from the timestamp
def extract_date_from_timestamp(timestamp):
    return timestamp.split("T")[0]

earthquake_data["date"] = earthquake_data['time'].apply(extract_date_from_timestamp)
print(earthquake_data)

#Trimming year month and day to 3 seperate columns
def extract_month_from_date(date):
    return date.split('-')[1]
def extract_year_from_date(date):
    return date.split('-')[0]
def extract_day_from_date(date):
    return date.split('-')[2]     
earthquake_data["month_of_occurence"] = earthquake_data['date'].apply(extract_month_from_date)
earthquake_data["day_of_occurence"] = earthquake_data['date'].apply(extract_day_from_date)
earthquake_data["year_of_occurence"] = earthquake_data['date'].apply(extract_year_from_date)
print(earthquake_data)
print(earthquake_data.isnull().sum())




# Visualization of tje latitutde and longitude of the location where the earthquake has happened 
fig, ax = plt.subplots(figsize=(15,8))
plt.plot(earthquake_data['longitude'], 
        earthquake_data['latitude'],
        linestyle='none', marker='.')
plt.suptitle("Earthquakes around the world")
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

## Visualize the magnitude around different places

sns.barplot(data=earthquake_data, x="place", y="mag")
plt.title('Magnitude around different places')
plt.xlabel('Place')
plt.ylabel('Magnitude')
plt.xticks(rotation=90)
plt.show()



#Taking magnitude value 3 as reference, we create 2 dataframes and visualize them

#df_highest is the dataframe with the places where the magnitude is greater than 3
df_highest = earthquake_data[earthquake_data["mag"]>3]
print(df_highest)

#df_lowest is the dataframe with the places where the magnitude is greater than 3
df_lowest =  earthquake_data[earthquake_data["mag"]<3]
print(df_lowest)

# Visualization for High Magnitude locations
sns.barplot(data=df_highest, x="place", y="mag")
plt.title('High Magnitude regions around the world')
plt.xlabel('Place')
plt.ylabel('Magnitude')
plt.xticks(rotation=90)
plt.show()

# Visualization for Low Magnitude locations
sns.barplot(data=df_lowest, x="place", y="mag")
plt.title('High Magnitude regions around the world')
plt.xlabel('Place')
plt.ylabel('Magnitude')
plt.xticks(rotation=90)
plt.show()



#Implementing on Random Forest Regressor.
scr = StandardScaler()
features = ['mag','depth','month_of_occurence','day_of_occurence','year_of_occurence']
target = ["latitude","longitude"]
X = earthquake_data[features]
y = earthquake_data[target]
#Splitting of Data set into Train and test using Stratified K fold spliting
X_train, X_test,y_train, y_test = train_test_split(X,y ,
                                random_state=245, 
                                test_size=0.25, 
                                shuffle=True)
X_train = scr.fit_transform(X_train)
X_test = scr.transform(X_test)
model1 =  MultiOutputRegressor(RandomForestRegressor(n_estimators = 196,random_state=0))
model1.fit(X_train,y_train)
prediction_values=model1.predict(X_test)
prediction1 = pd.DataFrame(prediction_values, columns=['latitude','longitude'])

worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
fig, ax = plt.subplots(figsize=(12, 6))
worldmap.plot(color="lightgrey", ax=ax)
x = prediction1['longitude']
y = prediction1['latitude']
plt.scatter(x, y,alpha=0.6,
            cmap='autumn')
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.savefig('output.png')
plt.show()

print("R2 score --> Random Forest ------->",r2_score(y_test,prediction_values))

print("MAE score --> Random Forest ------->",mean_absolute_error(y_test,prediction_values))
print("RSME score random forest ==>",math.sqrt(mean_squared_error(y_test,prediction_values)))

#Learning model for Support Vector 
svmModel =svm.LinearSVR( epsilon=1.0, tol=0.0002, C=1.0, loss='epsilon_insensitive',
 fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=3, max_iter=10000)
model2 =  MultiOutputRegressor(svmModel)
model2.fit(X_train,y_train)
prediction_values2 = model2.predict(X_test)
prediction2 = pd.DataFrame(prediction_values2, columns=['latitude','longitude'])
print("MAE score --> Support Vector Regression------->",mean_absolute_error(y_test,prediction_values2))
print("R2 score --> Support Vector Regressor ------->",r2_score(y_test,prediction_values2))
print("RSME score for svm ==>",math.sqrt(mean_squared_error(y_test,prediction_values2)))

worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
fig, ax = plt.subplots(figsize=(12, 6))
worldmap.plot(color="lightgrey", ax=ax)
x = prediction2['longitude']
y = prediction2['latitude']
plt.scatter(x, y,alpha=0.6,
            cmap='autumn')
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.savefig('output.png')
plt.show()

estimators = [  ('rfg',svmModel),  
        ('svr',RandomForestRegressor(n_estimators = 196,random_state=0))
        ]
stackreg =StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression())
stackreg=MultiOutputRegressor(stackreg)
stackreg.fit(X_train,y_train)
final_prediction = stackreg.predict(X_test)
final_prediction_df =pd.DataFrame(final_prediction, columns=['latitude','longitude']).to_csv('prediction_earthquake_1.csv')
final_prediction_df =pd.DataFrame(final_prediction, columns=['latitude','longitude'])
print("MAE score --> Stacking Regression------->",mean_absolute_error(y_test,final_prediction))
print("R2 score -->Stacking Regressor------->",r2_score(y_test,final_prediction))
print("RSME score Stacking Regressor ==>",math.sqrt(mean_squared_error(y_test,final_prediction)))

worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

fig, ax = plt.subplots(figsize=(12, 6))
worldmap.plot(color="lightgrey", ax=ax)
x = final_prediction_df['longitude']
y = final_prediction_df['latitude']
plt.scatter(x, y,alpha=0.6,
            cmap='autumn')
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.savefig('output.png')
plt.show()
    


