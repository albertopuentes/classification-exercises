import pandas as pd
# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer

# filter out warnings
import warnings
warnings.filterwarnings('ignore')

import acquire

# Use the function defined in acquire.py to load the iris data.
iris_df = acquire.new_iris_data()
iris_df.head()

# 2.Drop the species_id and measurement_id columns.
cols_to_drop = ['species_id']
df_iris_dropped = iris_df.drop(columns=cols_to_drop)
df_iris_dropped.head()

# 3) Rename the species_name column to just species.
df_iris_droppedr = df_iris_dropped.rename(columns={'species_name':'species'})
df_iris_droppedr.head()

# 4) Create dummy variables of the species name.
dummy_df = pd.get_dummies(df_iris_droppedr[['species']], dummy_na=False)
dummy_df.head()

#5 Create a function named prep_iris that accepts the untransformed iris data, and returns the data with the transformations above applied.
def prep_iris(df):
    # Drop the species_id and measurement_id columns.
    cols_to_drop = ['species_id']
    df_iris_dropped = iris_df.drop(columns=cols_to_drop)
    # Rename the species_name column to just species.
    df_iris_droppedr = df_iris_dropped.rename(columns={'species_name':'species'})
    # Create dummy variables of the species name.
    dummy_df = pd.get_dummies(df_iris_droppedr[['species']], dummy_na=False)
    # Combine my dfs to form a final cleaned version
    clean_iris_df = pd.concat([df_iris_droppedr, dummy_df], axis=1)
    return clean_iris_df

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.survived)
    return train, validate, test


def titanic_split(df):
    '''
    This function take in the titanic data acquired by get_titanic_data,
    performs a split and stratifies survived column.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.survived)
    return train, validate, test



def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column for
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test


def prep_titanic(df):
    '''
    This function take in the titanic data acquired by get_titanic_data,
    Returns prepped train, validate, and test dfs with embarked dummy vars,
    deck dropped, and the mean of age imputed for Null values.
    '''
    
    # drop rows where embarked/embark town are null values
    df = df[~df.embarked.isnull()]
    
    # encode embarked using dummy columns
    titanic_dummies = pd.get_dummies(df.embarked, drop_first=True)
    
    # join dummy columns back to df
    df = pd.concat([df, titanic_dummies], axis=1)
    
    # drop the deck column
    df = df.drop(columns='deck')
    
    # split data into train, validate, test dfs
    train, validate, test = titanic_split(df)
    
    # impute mean of age into null values in age column
    train, validate, test = impute_mean_age(train, validate, test)
    
    return train, validate, test