import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, MetaData, Table


def load_data(messages_filepath, categories_filepath):
    """
    Function loads data from two files; one with messages and one with the categories that
    the messages belong to. A column "id" is found in both files, linking messages and 
    categories. The data from the two files are merged into one Pandas DataFrame, where the
    category names are used as column headers
    
    Input:    messages_filepath: string, the path to the messages file
              categories_filepath: string, the path to the categories file
    Output:   df: DataFrame, a DataFrame holding messages and message categories
    """
    #Loading csv-files into DataFrames
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merging dataframes
    df = messages.merge(categories, on="id")
    
    #Splitting into categories
    categories = categories["categories"].str.split(";",expand=True)

    # select the first row of the categories dataframe
    # REF https://stackoverflow.com/questions/21385673/shortest-way-to-replace-parts-of-strings-in-numpy-array
    row = categories.head(1).values[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [cat.replace("-1","") for cat in [cat.replace("-0","") for cat in row]]
    #print(category_colnames)
    categories.columns = category_colnames
    categories.head()
    
    for column in categories:
        # set each value to be the last character of the string
        # REF https://stackoverflow.com/questions/33034559/how-to-remove-last-the-two-digits-in-a-column-that-is-of-integer-type
        categories[column] = categories[column].str[-1:]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #Drop old "categories" column
    df = df.drop('categories',axis=1)

    #Adding new columns for categories
    df = df.join(categories, on="id")

    return df
    #pass


def clean_data(df):
    """
    Function receives a DataFrame with data. Data are cleaned and then returned as a dataframe
    
    Input:    df: DataFrame, data that need to be cleaned
    Output:   df: DataFrame, after cleaning, data are returned as a dataframe
    """

    #Dropping duplicate rows
    df.drop_duplicates(inplace = True)

    #Drop all rows with NaN in column "related"
    df = df[pd.notnull(df['related'])]
    
    #drop all rows from category "related" not having value 0 or 1
    for key in df.related.value_counts().keys():
        if key > 1 or key < 0:
            #print(key)
            df = df[df.related != key]
    
    #Find the category columns and the number of unique values
    df_nunique = df.drop(['id', 'message', 'original', 'genre'],axis=1).nunique()
    
    #Drop the category columns having less than 2 values.
    col_to_remove = df_nunique[df_nunique < 2 ].dropna().index
    df = df.drop(col_to_remove, axis=1)
    
    #Converting all float64-columns in DataFrame to type int 
    #REF: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.select_dtypes.html
    #REF: https://stackoverflow.com/questions/21291259/convert-floats-to-ints-in-pandas
    df_cleaned = df.copy()
    for column in df_cleaned.select_dtypes(include=['float64']).columns:
       df_cleaned[column] = df_cleaned[column].astype(int)    
    
    return df_cleaned
    #pass


def save_data(df, database_filename):
    """
    Function receives a DataFrame with data and a filename. The data is then saved to the 
    filesystem in the form of a sqlite database file.
    
    Input:    df: DataFrame, data that need to be cleaned
              database_filename: string, the file name of the sqlite database file
    Output:   NA
    """
    table_name = 'DisasterResponseMessages'
    engine = create_engine('sqlite:///'+database_filename)

    try:
        meta = MetaData()
        sqlite_table_name = Table(table_name, meta)
        sqlite_table_name.drop(engine, checkfirst=False)
        #print(f"Table {table_name} deleted")
    except:
        #There will be an exception if you try to delete a non existing table
        #print(f"Exception when trying to delete table: {table_name}")
        None
    
    df.to_sql(table_name, engine, index=False)  
    return df

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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