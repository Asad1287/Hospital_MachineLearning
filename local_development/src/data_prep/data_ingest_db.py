"""
Loads data from csv to db using dask for parallel processing
1. Supports reading multiple csv files in a folder and combine them into one csv file and then comnvert to a dask dataframe
2. Supports reading a zip file and convert to a dask dataframe
3. Supports reading a csv, parquet, feather file and convert to a dask dataframe

"""

import pandas as pd 
import numpy as np 
import os
from dotenv import load_dotenv
import dask.dataframe as dd 
import glob
import sys
#"/workspaces/Hospital_BusinessCaseStudy/local_development/data/LengthOfStay.csv"
target = sys.argv[1]
type_data = sys.argv[2]
db_type = sys.argv[3]
load_dotenv()

if type_data == 'folder':
    """
    combine multiple csv files in a folder into one csv file
    """
    print("combining multiple csv files in a folder into one csv file")
    path = target
    all_files = glob.glob(os.path.join(path, "*.csv"))
    #load multiple csv files into a dask dataframe
    ddf = dd.read_csv(all_files)
elif type_data == 'zip':
    """
    unzip the zip file and load the csv file into a dask dataframe
    """
    print("unzip the zip file and load the csv file into a dask dataframe")
    ddf = dd.read_csv(target)
elif type_data == 'csv':
    """
    load the csv file into a dask dataframe
    """
    print("load the csv file into a dask dataframe")
    ddf = dd.read_csv(target)
elif type_data == 'feather':
    """
    load the feather file into a dask dataframe
    """
    print("load the feather file into a dask dataframe")
    ddf = dd.read_feather(target)
elif type_data == 'parquet':
    """
    load the parquet file into a dask dataframe
    """
    print("load the parquet file into a dask dataframe")
    ddf = dd.read_parquet(target)

else:
    print("data type not supported")
    sys.exit(1)


if db_type == 'postgres':

    try:
        postgres_user = os.getenv('postgres_user')
        postgres_pass = os.getenv('postgres_pass')
    except:
        print("postgres credentials not found")
        sys.exit(1)

if db_type == 'mysql':

    try:
        mysql_user = os.getenv('mysql_user')
        mysql_pass = os.getenv('mysql_pass')
    except:
        print("mysql credentials not found")
        sys.exit(1)


#set vdate as date format 
print("loading data from csv")


#replace rcount 5+ with 5 
ddf['rcount'] = ddf['rcount'].replace('5+','5')
#change rcount type to int
ddf['rcount'] = ddf['rcount'].astype(int)

if db_type == 'postgres':
    uri = f'postgresql+psycopg2://{postgres_user}:{postgres_pass}@localhost:5433/hospital'
elif db_type == "mysql":
    uri = f'mysql+pymysql://{mysql_user}:{mysql_pass}@localhost:3306/hospital'
elif db_type == "sqlite":
    uri = f'sqlite:///sqlite_db.db'
else:
    print("db type not supported")
    sys.exit(1)


print("loading data to db")
dd.to_sql(ddf,name = "TRAINDATA",uri=uri,if_exists='replace',index=False,parallel=True)
print("data loaded")