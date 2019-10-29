import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def etl_pipeline():
    """
    Reads in messages and categories csv. Merges files, cleans data and writes output to SQL Lite to be
    used in a ML model.
    """
    messages_df = pd.read_csv('messages.csv')
    categories_df = pd.read_csv('categories.csv')

    df = messages_df.merge(categories_df,on='id')

    categories = df.categories.str.split(pat=';',expand = True)

    #create column names for labeled data
    title_row = categories.iloc[0]
    category_colnames = title_row.apply(lambda x : x.split('-')[0]).tolist()
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] = categories[column].astype(int)

    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    df.loc[df.related == 2,'related'] = 1
    df = df.drop('child_alone',axis=1)

    engine = create_engine('sqlite:///DisasterResponseDatabase.db')
    df.to_sql('cleaned_disaster_data', engine, index=False)

etl_pipeline()