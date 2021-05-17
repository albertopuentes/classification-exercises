
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