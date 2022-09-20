#imports

import warnings
warnings.filterwarnings("ignore")
# tabular data stuff: numpy and pandas
import numpy as np
import pandas as pd
# data viz:
import matplotlib.pyplot as plt
import seaborn as sns

import env
import os


###-------------------------------------------acquisition--------------------------------------------------###
#ZILLOW DATASET
''' 
    This query pulls data from the zillow database from SQL.
    If this has already been done, the function will just pull from the zillow.csv
    '''
sql = '''
SELECT
    properties_2017.*,
    logerror, transactiondate, typeconstructiondesc, airconditioningdesc, architecturalstyledesc,
    buildingclassdesc, propertylandusedesc, storydesc, heatingorsystemdesc
FROM properties_2017
JOIN predictions_2017 ON properties_2017.parcelid = predictions_2017.parcelid
    AND predictions_2017.transactiondate LIKE '2017%%'
LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
LEFT JOIN airconditioningtype USING (airconditioningtypeid)
LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
LEFT JOIN buildingclasstype USING (buildingclasstypeid)
LEFT JOIN propertylandusetype USING (propertylandusetypeid)
LEFT JOIN storytype USING (storytypeid)
LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
    
JOIN (
    SELECT
        parcelid,
        MAX(transactiondate) AS date
    FROM predictions_2017
    GROUP BY parcelid
) AS max_dates ON properties_2017.parcelid = max_dates.parcelid
    AND predictions_2017.transactiondate = max_dates.date
    
WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

'''

#connection set ip
def conn(db, user=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#make function to acquire from sql
def new_zillow_data():
    df = pd.read_sql(sql,conn("zillow"))
    return df

def get_zillow_data():
    if os.path.isfile("zillow.csv"):
        #if csv is present locally, pull it from there
        df = pd.read_csv("zillow.csv", index_col = 0)
    else:
        #if not locally found, run sql querry to pull data
        df = new_zillow_data()
        df.to_csv("zillow.csv")
    return df

#IRIS DATASET

def new_iris_data():
    '''
    This function reads the iris data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT 
                    species_id,
                    species_name,
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                FROM measurements
                JOIN species USING(species_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    
    return df


def get_iris_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('iris_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('iris_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_iris_data()
        
        # Cache data
        df.to_csv('iris_df.csv')
        
    return df

###-----------------------------------------------preparation--------------------------------------------###

def summarize_column_nulls(df):

    '''
    this function takes in a dataframe of observations and attributes and returns a dataframe where each row
     is an atttribute name, the first column is the number of rows with missing values for that attribute, 
     and the second column is percent of total rows that have missing values for that attribute
    '''
    return pd.concat([
        df.isnull().sum().rename('num_rows_missing'),
        df.isnull().mean().rename('percent_rows_missing')
    ], axis = 1)



def summarize_row_nulls(df):

    '''
    this function removes any properties that are likely to be something other than single unit properties.
    '''
    return pd.concat([
        df.isnull().sum(axis = 1).rename('num_rows_missing'),
        df.isnull().mean(axis = 1).rename('percent_row_missing')
    ], axis = 1).value_counts().sort_index()


def handle_missing_values(df, prop_required_column, prop_required_row):
    '''
    this function handles the missing data and get rid of it, both in the columns and in the rows.
    then we can analyze the data
    '''
    print ('Before dropping nulls, %d rows, %d cols' % df.shape)
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1, thresh=n_required_column)
    #df = drop_columns(df)
    print('After dropping nulls. %d rows. %d cols' % df.shape)
    return df
