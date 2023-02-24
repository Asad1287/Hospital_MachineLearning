import pandas as pd
import joblib
from utils import *
from train_script import *
import sys 
import os
Test_Data_URL = sys.argv[1]
   
def test_production(Test_Data_URL:pd.DataFrame):
    df_raw = pd.read_csv(Test_Data_URL)
    
    df_raw = df_raw.drop(['eid','vdate','discharged','facid'],axis=1)
    #column_names.txt contains the column names of the dataset, it should match the column names in the dataset
    column_names = open('column_names.txt').read().splitlines()
    assert list(df_raw.columns) == column_names
       
    #read pickle model file
    model = joblib.load('model.pkl')
    #run inference on the test dataset and store results in column predictions and save file as predictions.csv
    predictions = model.predict(df_raw)
    df_raw['predictions'] = predictions
    
    df_raw.to_csv('predictions.csv')
      
if __name__ == '__main__':
    test_production(Test_Data_URL)