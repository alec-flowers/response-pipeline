import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(message_path,categories_path):
    """
    Load message & category data and merge on commmon id.

    Input:
        message_path (str) : path to message data
        categories_path (str) : path to categories data

    Output: 
        df (dataframe) : merged message & category data
    """
    messages_df = pd.read_csv(message_path)
    categories_df = pd.read_csv(categories_path)

    df = messages_df.merge(categories_df,on='id')

    return df

def etl_pipeline(df):
    """
    Cleans merged data by naming columns, extracting binary variables, and taking out duplicates.
    
    Input:
        df (datafrme) : Unclean dataframe

    Output:
        df (dataframe) : Cleaned dataframe
    """

    #Pull column names from data and label columns
    categories = df.categories.str.split(pat=';',expand = True)
    title_row = categories.iloc[0]
    category_colnames = title_row.apply(lambda x : x.split('-')[0]).tolist()
    categories.columns = category_colnames

    #turn string data into ints 
    for column in categories:
        categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] = categories[column].astype(int)

    #drop unecessary columns, duplicates, and correct non-binary data
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    df.loc[df.related == 2,'related'] = 1
    df = df.drop('child_alone',axis=1)

    return df

def save_data(df,database_filepath):
    """
    Create SQL Lite database to store the cleaned data.
    
    Input:
        df (dataframe)
        database_filepath (str)

    Output:
        sqlite database
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('cleaned_disaster_data', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = etl_pipeline(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()