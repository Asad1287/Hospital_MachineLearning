import unittest
import json
import pandas as pd
import sys
import numpy as np
import dask.dataframe as dd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib
from utils import *

def convert_to_training_vector(df_raw:pd.DataFrame) -> None:
    """
    create a base pipeline for datatransformation 
    """
    # drop the columns that are not needed if it contains the "eid", "vdate", "discharged", "facid"
    
    
    
    try :
        df_raw = df_raw.drop(["eid", "vdate", "discharged", "facid"], axis=1)
    except:
        pass 
    # convert the categorical columns to numerical columns
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder()
    )
    # create a column transformer
    preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object)),
    )
    # create a pipeline
    full_pipeline = Pipeline([("preprocessing", preprocessing)])
    # fit the data to full pipeline and pickle the pipeline
    full_pipeline.fit(df_raw)
    joblib.dump(full_pipeline, "full_pipeline.pkl")

def train_dataset(Data_URL: str, model_params: dict):
    """
    train a random forest model and save the model to a file using the model_params in the argument

    """
    # create a datapipeline
    # read the data from the csv file using dask dataframe
    df_raw = dd.read_csv(Data_URL)
    # drop the columns that are not needed
    target = df_raw.pop("lengthofstay")
    df_raw = df_raw.drop(["eid", "vdate", "discharged", "facid"], axis=1)
    # convert the categorical columns to numerical columns
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder()
    )
    # create a column transformer
    preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object)),
    )
    # create a pipeline
    full_pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("random_forest", RandomForestRegressor(random_state=42, **model_params)),
        ]
    )

    split_len = 0.2
    random_state = 42
    # split the data into train and test sets
    X_test = df_raw.head(int(len(df_raw) * split_len))
    X_train = df_raw.tail(int(len(df_raw) * (1 - split_len)))
    y_test = target.head(int(len(df_raw) * split_len))
    y_train = target.tail(int(len(df_raw) * (1 - split_len)))

    # train the model
    model = full_pipeline.fit(X_train, y_train)
    return model

data_url = sys.argv[1]
# write a unittest to test the train_dataset function, if the test fails, the pipeline will fail
class TestTrainDataset(unittest.TestCase):
    def test_train_dataset(self):
        df_raw = pd.read_csv(data_url)
        target = df_raw.pop("lengthofstay")
        df_raw = df_raw.drop(["eid", "vdate", "discharged", "facid"], axis=1)
        # column_names.txt contains the column names of the dataset, it should match the column names in the dataset
        column_names = open("column_names.txt").read().splitlines()
        self.assertEqual(list(df_raw.columns), column_names)
        X_train, X_test, y_train, y_test = split_data(df_raw, target)
        # get model_params from best_param.json file
        model_params = json.load(open("best_param.json"))
        model = train_dataset(data_url, model_params)
        self.assertGreater(model.score(X_train, y_train), 0.8)
        self.assertGreater(model.score(X_test, y_test), 0.8)
        # save the model to a file
        joblib.dump(model, "model.pkl")


if __name__ == "__main__":
    unittest.main()