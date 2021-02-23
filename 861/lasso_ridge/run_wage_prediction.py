import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics






dataset = pd.read_csv('wage_dataset.csv')

print(dataset)



region_dummies = pd.get_dummies(dataset['region'])
occ_dummies = pd.get_dummies(dataset['occ'])
## add polynomial terms
exp2 = dataset['exp'].astype(int)**2

data = pd.concat([
    dataset[['ability','age','female','education','exp']],
    region_dummies,
    occ_dummies,
    exp2
    ], axis = 1).values

# earnings
target = dataset.iloc[:, 1].values
print(target)


kfold_obj = KFold(n_splits = 4)
kfold_obj.get_n_splits(data)

test_case = 0
for training_index, test_index in kfold_obj.split(data):
    print(test_case)
    test_case += 1
    data_training = data[training_index]
    data_test = data[test_index]
    target_training = target[training_index]
    target_test = target[test_index]
    machine = linear_model.LinearRegression()
    machine.fit(data_training, target_training)
    new_target = machine.predict(data_test)
    R2_score = metrics.r2_score(target_test, new_target)
    print('R2 score:',R2_score)











