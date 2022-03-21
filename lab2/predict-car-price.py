import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

import math

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    
    def validate(self, y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)
        
    
    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X]) 

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y) 
    
        return w[0], w[1:] 
        

   
    def trainX(self, input, base): 
        df_num = input[base]                   
        df_num = df_num.fillna(0)
        return  df_num.values
        

def quest() :
    car_price = CarPrice()
    car_price.trim() 
    df = car_price.df

    np.random.seed(2) 
    n = len(df) 
    n_val, n_test, n_train = int(0.2 * n) , int(0.2 * n) ,n - (n_val + n_test) 
       
      

    index = np.arange(n) 
    np.random.shuffle(index) 

    df_shuffled = df.iloc[index] 
    
    
    df_train,  df_val, df_test = df_shuffled.iloc[:n_train].copy() , df_shuffled.iloc[n_train:n_train+n_val].copy() , df_shuffled.iloc[n_train+n_val:].copy()
     
     

   
   

    
    yTrain = np.log1p(df_train.msrp.values)
    yVal = np.log1p(df_val.msrp.values)
    yTest = np.log1p(df_test.msrp.values)

    base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity'] 
    xTrain = car_price.trainX(df_train, base) 

    
    w_0, w = car_price.linear_regression(xTrain, yTrain)

    
    X_val = car_price.trainX(df_val, base)
    y_pred_val = w_0 + X_val.dot(w)
    print("The rmse value of predicted MSRP and actual MSRP of validation set is ", car_price.validate(yVal, y_pred_val))

    
    X_test = car_price.trainX(df_test, base)
    y_pred_test = w_0 + X_test.dot(w)
    print("The rmse value of predicted MSRP and actual MSRP of test set is ", car_price.validate(yTest, y_pred_test))

    
    y_pred_MSRP_val = np.expm1(y_pred_val) 
    
    df_val['msrp_pred'] = y_pred_MSRP_val 
    
    print("Let us print out first 5 cars in our Validation Set's original msrp vs. predicted msrp")
    print(df_val.iloc[:,5:].head().to_markdown(), "\n")


if __name__ == "__main__":
    
    quest()
