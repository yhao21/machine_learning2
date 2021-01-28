
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics

dataset = pd.read_csv('dataset_3_outputs-1.csv')

target = dataset.iloc[:,2].values
data = dataset.iloc[:, 3:9].values


kfold_object = KFold(n_splits=4)

kfold_object.get_n_splits(data)


test_case = 0
for training_index, test_index in kfold_object.split(data):
    print(test_case)
    test_case = test_case + 1
    #print('training: ', training_index)
    #print('test: ', test_index)
    data_training = data[training_index]
    data_test = data[test_index]
    target_training = target[training_index]
    target_test = target[test_index]
    machine = linear_model.LogisticRegression()
    machine.fit(data_training, target_training)
    new_target = machine.predict(data_test)
    r2 = metrics.r2_score(target_test, new_target)
    print(r2)
    acc_score = metrics.accuracy_score(target_test, new_target)
    print(acc_score)
    conf_matrix = metrics.confusion_matrix(target_test, new_target)
    print(conf_matrix)



'''
conf_matrix:
                                target prediction
    actual target value
'''




'''
0.6479269941415051
0.5919558659464608
0.6331858815328153
0.6239990374375358

Normally we won't have a very high R2 for dummy variables, it is difficult.
'''



'''
0.6479269941415051
acc_score
0.912
0.5919558659464608
acc_score
0.898
0.6331858815328153
acc_score
0.9084
0.6239990374375358
acc_score
0.906


acc_score  = % of prediction == actual target. they should be exactly the same
Notice, for continuous Y variables, even if prediction is pretty close to actual
target value, acc_score would say it is not. 
So,

Use acc_score for dummy variables
Use R2 for continuous target variables
'''












