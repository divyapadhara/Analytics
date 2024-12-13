# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:49:49 2024

@author: shubh
"""

#loading libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
import os

import math 
from flask import Flask, request, render_template

app = Flask('__name__')

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

 #loading dataset into python
path = 'D:/Documents/shubh/Canada2023/CapeBretonUniversity/Semester4/CapstoneProject/Docsfromemployer/datasetupdated.xlsx'
df=pd.read_excel(path)
 
df=df.drop(['Promotion', 'Competitor_Price','Demand_Level','Initial_Stock','Product_ID','Product_Name'], axis=1)
selected_columns = ['Cost_Price', 'Market_Trend', 'Historical_Sales', 'Current_Stock',
       'Price', 'Category_Clothing', 'Category_Electronics',
       'Category_Home & Kitchen', 'Category_Sports', 'Season_Spring',
       'Season_Summer', 'Season_Winter']

@app.route('/')
def loadPage():
    return render_template('home.html', query="")


@app.route('/', methods=['POST'])
def marginprediction():
    # Load dataset and clean
    df = pd.read_excel(path)
    df = df.drop(['Promotion', 'Competitor_Price','Demand_Level','Initial_Stock','Product_ID','Product_Name', 'Margin'], axis=1)
    
    # Inputs from the form
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']

    # Create DataFrame from inputs
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7]]
    df1 = pd.DataFrame(data, columns=['Cost_Price', 'Market_Trend', 'Historical_Sales', 'Current_Stock', 'Price', 'Category', 'Season'])

    # Combine DataFrames
    df = pd.concat([df, df1], ignore_index=True)

    # Convert columns to numeric
    columns_to_convert = ['Cost_Price', 'Market_Trend', 'Historical_Sales', 'Price', 'Current_Stock']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    # Check for NaN values
    if df[columns_to_convert].isna().sum().any():
        print("Warning: NaN values found after conversion.")

    # One-hot encoding and selecting columns
    df3 = pd.get_dummies(df, drop_first=True)
    df3 = df3[selected_columns]

    # Predict using the model
    o1 = loaded_model.predict(df3.iloc[-1,:].values.reshape(1, -1))[0]
    o2 = o1*100
    o2 = round(o2,2)

    return render_template('home.html', output=o2, 
                           query1=request.form['query1'], 
                           query2=request.form['query2'],
                           query3=request.form['query3'],
                           query4=request.form['query4'],
                           query5=request.form['query5'], 
                           query6=request.form['query6'], 
                           query7=request.form['query7'])


app.run(port=9000)



