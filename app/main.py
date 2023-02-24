from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import numpy as np


from sklearn.base import BaseEstimator

app = FastAPI()
import pandas as pd 
import joblib
""""
#the following script recieves the following fields inputs
#rcount
gender
dialysisrenalendstage
asthma
irondef
pneum
substancedependence
psychologicaldisordermajor
depress
psychother
fibrosisandother
malnutrition
hemo
hematocrit
neutrophils
sodium
glucose
bloodureanitro
creatinine
bmi
pulse
respiration
secondarydiagnosisnonicd9

# 
#  run unpickles the model.pkl file, runs inference on LengthOfStay_Prod.csv and returns the predictions
"""

class HospitalData(BaseModel):
    rcount:str 
    gender:str 
    dialysisrenalendstage:int 
    asthma:int
    irondef:int
    pneum:int
    substancedependence:int
    psychologicaldisordermajor:int
    depress:int
    psychother:int
    fibrosisandother:int
    malnutrition:int
    hemo:int
    hematocrit:float
    neutrophils:float
    sodium:float
    glucose:float
    bloodureanitro:float
    creatinine:float
    bmi:float
    pulse:float
    respiration:float
    secondarydiagnosisnonicd9:int

def get_model_from_pickle(model_path: str) -> BaseEstimator:
    model = joblib.load(model_path)
    return model

#valdidation methods , input string must be either M or F
def string_gender_validation(gender_string:str):
    "must be either M or F"
    if not (gender_string == "M" or gender_string == "F"):
        return False
    else:
        return True

def string_rcount_validation(rcount_string:str):
    "must be either M or F"
    possible_values = ['0','1','2','3','4','5+']
    if not (rcount_string in possible_values):
        return False
    else:
        return True


#input must be either 0 or 1

def binary_validation(binary_value:int):
    "must be either 0 or 1"
    if not (binary_value == 0 or binary_value == 1):
        return False

    else:
        return True


@app.get("/")
def read_root():
    
    return {"Welcome to the LOS prediction service API"}

@app.post("/predict")
def predict( hospital_data: HospitalData):
    rcount = hospital_data.rcount
    gender = hospital_data.gender
    dialysisrenalendstage = hospital_data.dialysisrenalendstage
    asthma = hospital_data.asthma
    irondef = hospital_data.irondef
    pneum = hospital_data.pneum
    substancedependence = hospital_data.substancedependence
    psychologicaldisordermajor = hospital_data.psychologicaldisordermajor
    depress = hospital_data.depress
    psychother = hospital_data.psychother
    fibrosisandother = hospital_data.fibrosisandother
    malnutrition = hospital_data.malnutrition
    hemo = hospital_data.hemo
    hematocrit = hospital_data.hematocrit
    neutrophils = hospital_data.neutrophils
    sodium = hospital_data.sodium
    glucose = hospital_data.glucose
    bloodureanitro = hospital_data.bloodureanitro
    creatinine = hospital_data.creatinine
    bmi = hospital_data.bmi
    pulse = hospital_data.pulse
    respiration = hospital_data.respiration
    secondarydiagnosisnonicd9 = hospital_data.secondarydiagnosisnonicd9
    
    
    if not binary_validation(dialysisrenalendstage):
        return {"error": "dialysisrenalendstage must be either 0 or 1"}
    
    if not binary_validation(asthma):
        return {"error": "asthma must be either 0 or 1"}
    
    if not binary_validation(irondef):
        return {"error": "irondef must be either 0 or 1"}
    
    if not binary_validation(pneum):
        return {"error": "pneum must be either 0 or 1"}
    
    if not binary_validation(substancedependence):
        return {"error": "substancedependence must be either 0 or 1"}
    
    if not binary_validation(psychologicaldisordermajor):
        return {"error": "psychologicaldisordermajor must be either 0 or 1"}
    
    if not binary_validation(depress):
        return {"error": "depress must be either 0 or 1"}
    
    if not binary_validation(psychother):
        return {"error": "psychother must be either 0 or 1"}
    
    if not binary_validation(fibrosisandother):
        return {"error": "fibrosisandother must be either 0 or 1"}
    
    if not binary_validation(malnutrition):
        return {"error": "malnutrition must be either 0 or 1"}
    
    if not binary_validation(hemo):
        return {"error": "hemo must be either 0 or 1"}
    
    if not string_gender_validation(gender):
        return {"error":"check gender values"}
    
    if not string_rcount_validation(rcount):
        return {"error":"check rcount values"}

    
    


    model = get_model_from_pickle('model.pkl')
    #convert the inputs to a array 
    col_names = ["rcount",'gender','dialysisrenalendstage','asthma','irondef','pneum','substancedependence','psychologicaldisordermajor','depress','psychother','fibrosisandother','malnutrition','hemo','hematocrit','neutrophils','sodium','glucose','bloodureanitro','creatinine','bmi','pulse','respiration','secondarydiagnosisnonicd9']
    X_test = [rcount,gender,dialysisrenalendstage,asthma,irondef,pneum,substancedependence,psychologicaldisordermajor,depress,psychother,fibrosisandother,malnutrition,hemo,hematocrit,neutrophils,sodium,glucose,bloodureanitro,creatinine,bmi,pulse,respiration,secondarydiagnosisnonicd9]
    #X_test = np.array(['0' ,'M', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10.1, 6.4, 135.0401112 ,141.7393706 ,29.0 ,0.65148965, 28.35099534, 38.89135671 ,6.7 ,4])
    df_raw = pd.DataFrame([X_test], columns=col_names)

    predictions = model.predict(df_raw)
    return {"prediction": predictions.tolist()}


