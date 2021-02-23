
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from math import radians



dataset = pd.read_csv('chicago_crime_2018.csv')
print(dataset)

dataset = dataset.sample(frac = 0.1).reset_index(drop=True)
print(dataset)

'''
Relative Location
The relative distance is more meaningful compare to the direct value of distance
Example, relative distance to police station


Euclidean distance
use Cartesian theorem is not meaningful especially when distance is long.
Notice, earth, not flat. The distance is a curve


we need Haversine Distance
'''

def get_haversine(x):
    '''
    haversine requires radians
    '''
    lat1 = x['Latitude']
    long1 = x['Longitude']

    ## base center. btw, this is the location of Trump tower
    lat2 = 41.8889
    long2 = -87.6264

    # convert to radian
    loc1 = [radians(lat1), radians(long1)]
    loc2 = [radians(lat2), radians(long2)]

    ## we need to multiply the radius of the earth in meters
    ## radius of the earth is 6357KM
    return (haversine_distances([loc1, loc2]) *6357000)[0][1]


## we what to apply this to all rows in dataset, row = axis = 1
distance_results = dataset.apply(get_haversine, axis = 1)
print(distance_results)
dataset['distance_from_downtown'] = distance_results



'''
One technique, 
you can chop a city into many square area by restricting the value of latitude
and longitude, e.g. lag in (41.1, 44), long in (-87, 89)
'''





