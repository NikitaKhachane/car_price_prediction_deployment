import pandas as pd 
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from pandas import set_option
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")
set_option('precision', 2)
from sklearn.metrics import  mean_squared_error , r2_score



# importing csv file
df = pd.read_csv("car_sales")


# encoding catagorical values
from sklearn import preprocessing 


# for fule type column
fit = preprocessing.LabelEncoder()
fit.fit(['CNG', 'Diesel' ,'Petrol', 'LPG' ,'Electric'])
df['Fuel_Type'] = fit.transform(df.Fuel_Type)
#print(df.Fuel_Type.unique())


# for transmission column
fit.fit(['Manual', 'Automatic'])
df['Transmission'] = fit.transform(df.Transmission)
#print(df.Transmission.unique())


# "name" column
fit.fit(['Maruti' ,'Hyundai', 'Honda', 'Audi' ,'Nissan', 'Toyota' ,'Volkswagen', 'Tata',
 'Land', 'Mitsubishi', 'Renault' ,'Mercedes-Benz', 'BMW', 'Mahindra', 'Ford',
 'Porsche', 'Datsun' ,'Jaguar' ,'Volvo', 'Chevrolet', 'Skoda', 'Mini', 'Fiat',
 'Jeep', 'Smart' ,'Ambassador', 'Isuzu', 'ISUZU' ,'Force' ,'Bentley',
 'Lamborghini'])
df['Name'] = fit.transform(df.Name)
#print(df.Name.unique())


# for owner type column 
for i in df.index:
    if df.loc[i,"Owner_Type"] == "First":
        df.loc[i,"Owner_Type"] = 1
    if df.loc[i,"Owner_Type"] == "Second":
        df.loc[i,"Owner_Type"] = 2
    if df.loc[i,"Owner_Type"] == "Third":
        df.loc[i,"Owner_Type"] = 3
    if df.loc[i,"Owner_Type"] == "Fourth & Above":
        df.loc[i,"Owner_Type"] = 4
        
        
df.drop('Year',inplace = True,axis = 1)

df.drop('Location',inplace = True,axis = 1)

df.Owner_Type = df.Owner_Type.astype('int32')

df.Age = df.Age.astype('int32')




# model muilding - 
df.drop("Unnamed: 0",axis=1, inplace = True)
x = df.drop("Price",axis=1)   # independent variable
y = df.Price                  # dependent variablete

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 50)


# instantiate the model
rf = RandomForestRegressor()

# fit the model
rf.fit(x_train,y_train)


# make pickel file of our model
pickle.dump(rf, open("model.pkl", "wb"))


















