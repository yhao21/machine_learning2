import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz






'''
Use random forest in time series data.
'''


dataset = pd.read_csv('temperature_data.csv')

dataset = pd.get_dummies(dataset)


target = dataset['actual'].values
## axis = 1 specifice actual is a column
data = dataset.drop('actual', axis = 1).values






kfold_object = KFold(n_splits = 4)
kfold_object.get_n_splits(data)


test_case = 0
for training_index, test_index in kfold_object.split(data):
    print(test_case)
    test_case = test_case + 1

    data_training = data[training_index]
    data_test = data[test_index]
    target_training = target[training_index]
    target_test = target[test_index]

    machine = RandomForestRegressor(n_estimators = 201,max_depth = 30)
    machine.fit(data_training, target_training)
    new_target = machine.predict(data_test)
    MAE = metrics.mean_absolute_error(target_test, new_target)
    print(MAE)



machine = RandomForestRegressor(n_estimators = 201,max_depth = 30)
machine.fit(data,target)

feature_list = dataset.drop('actual', axis = 1).columns
feature_importances_raw = list(machine.feature_importances_)
print(feature_importances_raw)

feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(feature_list, feature_importances_raw)]

#for item in feature_importances:
#    print(item)


### we want to sorted by the value rather feature name. 
### example: x is ('index', 0.029) ... x[1] is 0.029
### reverse says that we want the max value come up first
feature_importances = sorted(feature_importances, key = lambda x:x[1], reverse=True)

### one line print 
### :13 means lave 13 character space from the start of the line
### to the second item
[print('{:13} {}'.format(*i)) for i in feature_importances]



'''

temp_lag1     0.526
historical    0.371
index         0.028
temp_lag2     0.023
day           0.022
month         0.005
week_Mon      0.005
week_Sat      0.004
week_Sun      0.004
week_Fri      0.003
week_Tues     0.003
week_Wed      0.003
week_Thurs    0.002
year          0.0


Now we know the most importance feature is yesterday's temperature and
the historical average of today's temperature.
'''

print(feature_importances_raw)

x_values = list(range(len(feature_importances_raw)))
print(x_values)
## label x axis
## rotation ask plt show feature name vertically
plt.xticks(x_values, feature_list, rotation = 'vertical')

plt.ylabel('importance')
plt.xlabel('features')
plt.title('Feature Importance')
## it will fit the picture size to its content
## you won't have any words out of the figure
plt.tight_layout()


plt.bar(x_values, feature_importances_raw, orientation = 'vertical')
plt.show()

'''
sklearn save your trees into a file, and you need graphviz to ploy the tree graph
'''




