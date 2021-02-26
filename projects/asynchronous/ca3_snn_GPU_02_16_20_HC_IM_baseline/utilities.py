# Import libraries that are necessary for the function library
import pandas as pd
import numpy as np


# Function to insert row in the dataframe
def Insert_row_(row_number, df, row_value):
    # Slice the upper half of the dataframe
    df1 = df[0:row_number]
    # Store the result of lower half of the dataframe
    df2 = df[row_number:]
    # Inser the row in the upper half dataframe
    df1.loc[row_number]=row_value
    # Concat the two dataframes
    df_result = pd.concat([df1, df2])
    # Reassign the index labels
    df_result.index = [*range(df_result.shape[0])]
    # Return the updated dataframe
    return df_result

# Function to get the name of a dataframe
def get_df_name(df):
    name = [x for x in globals() if globals()[x] is df][0]
    return name

# Function to rename Hippocampome cell types names to a syntax amenable to
# CARLsim4
def cellTypeNameCARL(df, dfName = None, colName = None, newName = None):
    if any(x in dfName for x in ("SynWeight","STP","NumContacts","Multiplier")):
        df.rename(columns = {list(df)[0]:newName}, inplace='True');
        df[newName] = df[newName].str.replace(" ", "_"). \
                      str.replace("+", "").str.replace("-", "_");
        df.columns = [c.replace(' ', '_').replace("+", ""). \
                      replace("-", "_").replace("(","").replace(")","") \
                      for c in df.columns];
    elif any(x in dfName for x in ("ConnMat","ConnMatE2I")):
        df[colName] = df[colName].str.replace(" ", "_"). \
                      str.replace("+", "").str.replace("-", "_");
        df.columns = [c.replace(' ', '_').replace("+", ""). \
                      replace("-", "_").replace("(","").replace(")","") \
                      for c in df.columns];
    else:
        df[colName] = df[colName].str.replace(" ", "_"). \
                      str.replace("+", "").str.replace("-", "_");
