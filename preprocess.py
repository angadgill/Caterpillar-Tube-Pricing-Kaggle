__author__ = 'angad'
import numpy as np
import pandas as pd


def scale(x):
    x = np.array(x)
    return (x - x.mean())/x.std()
    # return x/x.std()


def long_to_wide(df, index, column, remove_value='NONE'):
    # Converts unique values in the 'column' column
    # to multiple columns and fills 0s and 1s in the cells
    # Also drops the 'column' from the df returned
    # removes rows with remove value in 'column'
    temp = df[[index, column]]
    temp = temp.drop_duplicates()
    temp = temp[temp[column] != remove_value]
    temp['values'] = 0
    temp = temp.pivot_table(index=index, columns=column, values='values')
    temp.columns = column + '_' + temp.columns.astype(str)
    temp = temp.reset_index()
    temp = temp.replace(0, 1)
    temp = temp.fillna(0)
    df = pd.merge(df, temp, on=index, how='left')
    df = df.fillna(0) #need to do this to avoid NaNs created by the merge operation
    df = df.drop(column, axis=1)
    return df
