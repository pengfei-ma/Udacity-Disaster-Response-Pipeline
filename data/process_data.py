import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

def load_data(messages_filepath, categories_filepath):
    # read the csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge the datasets by common id
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    categories = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[[1]]
    # rename the column by 'row'
    category_colnames = [i.split("-")[0] for i in row.iloc[0]]
    categories.columns = category_colnames

    # Convert category values to just number 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis = 1)
    # replace categories column in df with new category columns
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df = df.drop_duplicates()

    return df


def to_database(db_name, df):
    engine = create_engine('sqlite:///' + db_name)
    table_name = db_name.replace(".db","") + "_table"
    print(table_name)
    df.to_sql(table_name, engine, index=False, if_exists='replace')

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        to_database(database_filepath, df)
        
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