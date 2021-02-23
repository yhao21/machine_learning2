import pandas as pd
from datetime import datetime
import numpy as np




dataset = pd.read_csv('chicago_crime_2018.csv')
print(dataset)


## to save our time, we can use sample dataset to test to_datetime function

## frac = 0.1 means that the program will randomly drop 90% data. Left with only
## 10% data

dataset = dataset.sample(frac = 0.1).reset_index(drop=True)




date = dataset['Date']
print(date)

## pandas builtin function to_datetime transfer time data to python datatime form
dataset['Date'] = pd.to_datetime(dataset['Date'])
print(dataset['Date'])



# extract year info from datetime
dataset['year'] = dataset['Date'].dt.year
# extract month info from datetime
dataset['month'] = dataset['Date'].dt.month
# extract hour info from datetime
dataset['hour'] = dataset['Date'].dt.hour
dataset['minute'] = dataset['Date'].dt.hour




## regroup hours to different groups, i.e., group 0 = 0 AM to 3 AM
dataset['hour_slot'] = np.select([
    (dataset['hour'] < 4),
    (dataset['hour'] < 8),
    (dataset['hour'] < 12),
    (dataset['hour'] < 16),
    (dataset['hour'] < 20),
    (dataset['hour'] < 24),
    ],[0,1,2,3,4,5])



dataset['minute_slot'] = np.select([
    (dataset['minute'] < 15),
    (dataset['minute'] < 30),
    (dataset['minute'] < 45),
    (dataset['minute'] < 60),
    ],[0,1,2,3])


'''
how many minutes have passed, i.e., 5:40 am has 5*4 =10 15-minutes and 40 is
in the third 15 minutes slot
so, 5:40 is in  20+3=23 slot
'''
dataset['minutes_slot_in_date'] = dataset['hour'] * 4 + dataset['minute_slot']


print(dataset[['Date','hour', 'hour_slot']])


## compare datetime difference, i.e., compare to 2018-1-24 0:0:0 AM
dataset['diff'] = dataset['Date'] - datetime(2018,1,24,0,0,0)
print(dataset['diff'])



## count # of crime within a time slot
## start from 0, end with 95, step = 1
## there are 96 15-mins in a day
time_block_dataset = pd.DataFrame()
time_block_dataset['time_in_day'] = np.arange(0,96,1)
print(time_block_dataset)

## check if there are crimes happening during midnight (first 15 mins)
crime_counts = np.sum(dataset['minutes_slot_in_date'] == 0)
print(crime_counts)

'''
trick:
    you can convert True and False to 1 and 0 by multiply the upper code by 1
'''
print((dataset['minutes_slot_in_date'] == 0)*1)











## time data are categorical, we must transfer them to dummies
month_data = pd.get_dummies(dataset['month'])
## notice, we normally don't use each hour when doing time fixed effect,
