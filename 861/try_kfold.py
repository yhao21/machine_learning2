import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics
from matplotlib import pyplot as plt




dataset = pd.read_csv('dataset_3_outputs-1.csv')

target = dataset.iloc[:,0].values
data = dataset.iloc[:, 3:9].values


kfold_object = KFold(n_splits=4)

kfold_object.get_n_splits(data)


test_case = 0
for training_index, test_index in kfold_object.split(data):
    print(test_case)
    test_case = test_case + 1
    print('training: ', training_index)
    print('test: ', test_index)
    data_training = data[training_index]
    data_test = data[test_index]
    target_training = target[training_index]
    target_test = target[test_index]
    machine = linear_model.LinearRegression()
    machine.fit(data_training, target_training)
    new_target = machine.predict(data_test)
    r2 = metrics.r2_score(target_test, new_target)
    print(r2)
    new_target_newaxis = new_target[:,np.newaxis]
    results_machine = linear_model.LinearRegression()
    results_machine.fit(new_target_newaxis, target_test)
    plt.scatter(new_target_newaxis, target_test)
    plt.plot(new_target_newaxis, results_machine.predict(new_target_newaxis),c = 'red')
    plt.savefig('scatterplot' + str(test_case) + '.png')
    plt.clf()




