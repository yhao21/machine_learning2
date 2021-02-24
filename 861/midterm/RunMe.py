import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import haversine_distances
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from math import radians


df = pd.read_csv('AB_NYC_2019.csv')

####===================
#### check missing values, and delete these rows

df_missing_count = df.isnull().sum()
#print(df_missing_count)
#print(df.head(5))
'''
missing:
name                                 16
host_name                            21
last_review                       10052
reviews_per_month                 10052
'''
df_missing = df.isnull()
## find index of those rows with missing values, should in list
df_indexing = df_missing[(df_missing.last_review == True) | (df_missing.reviews_per_month == True)\
        | (df_missing.name == True) | (df_missing.host_name == True)].index.tolist()
#print(df_indexing)
## delete these rows
df = df.drop(index = df_indexing)


print(df)

print(df.columns.tolist())

df = df[['price','neighbourhood_group','latitude', 'longitude', 'room_type','minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']]
target = df['price'].values

print(df)
print(df['room_type'])

####=================
#### use 2021/02/24 as base, compute how long (days) there's no new review.
now_date = datetime(2021,2,24)
df['Date'] = pd.to_datetime(df['last_review'])
df['last_review_date_diff'] = -(df['Date'] - now_date).apply(lambda x: x.days)
print(df)


####=================
#### compute the distance to the center of NYC, let's use Empire State Building as benchmark
#### lat and logi for Empire State Building: 40.748440 (lat) -73.985664 (log)

def get_haversine(x):
    '''
    haversine requires radians
    '''
    lat1 = x['latitude']
    long1 = x['longitude']

    ## base center. 
    lat2 = 40.748440
    long2 = -73.985664

    # convert to radian
    loc1 = [radians(lat1), radians(long1)]
    loc2 = [radians(lat2), radians(long2)]

    ## we need to multiply the radius of the earth in meters
    ## radius of the earth is 6357KM
    return (haversine_distances([loc1, loc2]) *6357000)[0][1]

df['distance_from_empire'] = df.apply(get_haversine, axis = 1)
print(df)


####====================
#### get dummies
room_dummies = pd.get_dummies(df['room_type'])
neighbour_dummies = pd.get_dummies(df['neighbourhood_group'])
print(neighbour_dummies)



####====================
#### get clean df
print(df.columns.tolist())
clean_df = df.drop(['neighbourhood_group','room_type', 'last_review', 'latitude', 'longitude',\
        'Date'], axis = 1)
clean_df = pd.concat([
    clean_df,
    room_dummies,
    neighbour_dummies], axis = 1)


print(clean_df)


target = clean_df.iloc[:, 0].values
data = clean_df.iloc[:, 1:].values

#machine = linear_model.Ridge(alpha = 0.001, normalize=True)
machine = linear_model.Lasso(alpha = 0.001, normalize=True)
#machine = linear_model.LinearRegression()



def run_kfold(split_number,data,target,machine):

    kfold_object = KFold(n_splits=split_number)
    kfold_object.get_n_splits(data)
    
    results = []

    for training_index,test_index in kfold_object.split(data):
        data_training,data_test = data[training_index],data[test_index]
        target_training,target_test =target[training_index],target[test_index]
        machine.fit(data_training,target_training)
        prediction = machine.predict(data_test)
        R2_score = metrics.r2_score(target_test, prediction)
        results.append(R2_score)


    return results



results = run_kfold(4,data,target,machine)
print(results)








