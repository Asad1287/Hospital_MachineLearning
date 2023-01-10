import pandas as pd 
import numpy as np 
import os
from dotenv import load_dotenv
import dask.dataframe as dd 
import sys
filename = sys.argv[1]
load_dotenv()

postgres_user = os.getenv('postgres_user')
postgres_pass = os.getenv('postgres_pass')
from sqlalchemy import create_engine

#set vdate as date format 
print("loading data from csv")
ddf = dd.read_csv(filename , parse_dates=['vdate','discharged'])

#replace rcount 5+ with 5 
ddf['rcount'] = ddf['rcount'].replace('5+','5')
#change rcount type to int
ddf['rcount'] = ddf['rcount'].astype(int)



uri = f'postgresql+psycopg2://{postgres_user}:{postgres_pass}@localhost:5433/hospital'
#engine = create_engine(
#    f'postgresql+psycopg2://{postgres_user}:{postgres_pass}@localhost:5433/hospital')

#create table and load data into postgres 
dd.to_sql(ddf,name = "TRAINDATA",uri=uri,if_exists='replace',index=False,parallel=True)
print("data loaded")