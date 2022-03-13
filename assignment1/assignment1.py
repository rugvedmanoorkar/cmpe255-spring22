import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


df = pd.read_csv(r"/Users/rugvedmanoorkar/Documents/SJSU/Sem 2/Levels_Fyi_Salary_Data.csv")
df.head()

df['company'] = df['company'].fillna("NA")
df.drop_duplicates(inplace=True)
df.drop(["timestamp"], axis = 1, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df['year'] = df['timestamp'].dt.year



df = df.drop(columns = ['Doctorate_Degree', 'Race_Asian', 'Race_White', 'Race_Two_Or_More', 'level', 'tag','Race', 'Education', 'cityid', 'otherdetails', 'Some_College', 'Masters_Degree', 'Bachelors_Degree', 
                        'Race_Black', 'Race_Hispanic', 'Highschool', 'rowNumber', 'dmaid', 'gender'])







labelEncoder = LabelEncoder()
df['company'] = labelEncoder.fit_transform(df['company'])

labelEncoderLocation = LabelEncoder()
df['loc'] = labelEncoderLocation.fit_transform(df['location'])

labelencoderTitle = LabelEncoder()
df['title'] = labelencoderTitle.fit_transform(df['title'])



X = df.iloc[:, :-1]
y = df.iloc[:, -1]

k = 0.9

X_train = X.sample(frac = k, random_state = 54)
y_train = y.sample(frac = k, random_state = 54)
X_test = X.drop(X_train.index)
y_test = y.drop(y_train.index)

model = Sequential()
model.add(Dense(15, input_dim = len(X_train.iloc[0]), activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

model.compile(loss='sparse_categorical_crossentropy', optimizer = "adam", metrics = 'mean_squared_error')



y_pred = model.predict(X_test)
y_pred_flat = y_pred.flatten()

## Q1
pos = 'Software Developer'
exp = 5
loc = 'San Jose, CA'

def Q1(pos, exp, loc):
  yearsatcompany, bonus = 5, 34000
  basesalary, stockgrantvalue = 244000, 70000
  year = 2021
  company = 'Microsoft'

  pos = labelencoderTitle.transform([pos])[0]
  loc = labelEncoderLocation.transform([loc])[0]
  temp = labelEncoder.transform([company])[0]
  df = pd.DataFrame({'year': [year],'company': [temp],'position': [pos],'location': [loc],'experience': [exp], "yearsatcompany": [yearsatcompany], 
            'basesalary': [basesalary], 'stockgrantvalue': [stockgrantvalue], 'bonus': [bonus]})
  print(' Salary: ' ,model.predict(df).flatten()[0])
  



## Q2


company = 'Microsoft'
pos = 'Software Developer'
exp = 5
loc = 'San Jose, CA'

def Q2(company, pos, exp, loc):
  yearsatcompany, bonus = 7, 60000
  basesalary, stockgrantvalue = 766000, 90000  
  year = 2023
  
  pos = labelencoderTitle.transform([pos])[0]
  loc = labelEncoderLocation.transform([loc])[0]
  temp = labelEncoder.transform([company])[0]

  
  df = pd.DataFrame({'year':[year],'company':[temp],'pos':[pos],'loc':[loc],'exp':[exp], "yearsatcompany": [yearsatcompany], 
            'basesalary': [basesalary], 'stockgrantvalue': [stockgrantvalue], 'bonus': [bonus]})
  
  print('Salary: ' ,model.predict(df).flatten()[0])






yearsatcompany = 5


exp, loc = 3, 'San Jose, CA'
bonus, year = 60000, 2022
basesalary, stockgrantvalue = 155000, 20000

tempComp = labelEncoder.transform([company])[0]
pos = labelencoderTitle.transform([pos])[0]
loc = labelEncoderLocation.transform([loc])[0]


data = {'year':[year],'company':[tempComp],'pos':[pos],'loc':[loc],'exp':[exp], "yearsatcompany": [yearsatcompany], 
          'basesalary': [basesalary], 'stockgrantvalue': [stockgrantvalue], 'bonus': [bonus]}
df = pd.DataFrame(data)
y_test_pred_sal = model.predict(df).flatten()
print('Predicted Salary: ', y_test_pred_sal[0])

