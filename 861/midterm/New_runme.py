
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import haversine_distances
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from math import radians
from matplotlib import pyplot as plt



class CleanData():

    def __init__(self, df):
        self.df = df

    
    def drop_missing_values(self):
        '''
        check missing values, and delete these rows

        missing:
        name                                 16
        host_name                            21
        last_review                       10052
        reviews_per_month                 10052
        '''

        df_missing_count = df.isnull().sum()
        df_missing = df.isnull()
        ## find index of those rows with missing values, should in list
        df_indexing = df_missing[(df_missing.last_review == True) | (df_missing.reviews_per_month == True)\
                | (df_missing.name == True) | (df_missing.host_name == True)].index.tolist()
        #print(df_indexing)
        ## delete these rows
        df = df.drop(index = df_indexing)







if __name__ == '__main__':
    df = pd.read_csv('AB_NYC_2019.csv')
    CleanData(df).drop_missing_values()
