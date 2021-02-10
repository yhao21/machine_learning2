import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics

'''
                                occ, ability, age, ...
earning missing         ?               x

earning not missing     x               x


we
'''






dataset_missing = pd.read_csv('dataset_missing.csv', dtype = 'object')


print(dataset_missing)
### isnull() check if there are missing data in the dataframe
num_missing = dataset_missing.isnull().sum()
print(num_missing)

'''
    id              0
    earnings     5412
    occ             0
    ability         0
    age             0
    region          0
    female          0
    education       0
    exp             0
    year            0
we have 5412 missing data, normally, if missing value > 20% you should not use 
this variable.
'''



#for i in range(2000,2011):
#    print('\nyear %d' % i)
#    print(dataset_missing[dataset_missing['year'] == str(i)].isnull().sum())

###a = dataset_missing[dataset_missing['earnings'].notnull()].values
###print(a)

def data_imputation(dataset, impute_target_name, impute_data):
    impute_target = dataset[impute_target_name]
    ### axis = 1 means horizontally merge dfs
    subdataset = pd.concat([impute_target, impute_data],axis = 1)
    data = subdataset.loc[:,subdataset.columns != impute_target_name][subdataset[impute_target_name].notnull()].values
    target = dataset[impute_target_name][subdataset[impute_target_name].notnull()].values
    kfold_obj = KFold(n_splits=4)
    kfold_obj.get_n_splits(data)

    i = 0
    for training_index, test_index in kfold_obj.split(data):
        print("case: %d" % i)
        data_training = data[training_index]
        target_training = target[training_index]
        data_test = data[test_index]
        target_test = target[test_index]
        one_fold_machine = linear_model.LinearRegression()
        one_fold_machine.fit(data_training, target_training)
        new_target = one_fold_machine.predict(data_test)
        MAE = metrics.mean_absolute_error(target_test, new_target)
        print(MAE)
        i += 1

    all_variables = subdataset.loc[:,subdataset.columns != impute_target_name].values
    machine = linear_model.LinearRegression()
    machine.fit(data, target)
    results = machine.predict(all_variables)

    return results




### form the region dummy
region_dummies = pd.get_dummies(dataset_missing['region'])
occ_dummies = pd.get_dummies(dataset_missing['occ'])


impute_data = dataset_missing[['ability','age','female','education','exp']]
#impute_data = dataset_missing[['female']]


impute_data = pd.concat([impute_data, region_dummies, occ_dummies], axis = 1)

new_earnings = pd.DataFrame(data_imputation(dataset_missing, 'earnings', impute_data))
### inplace = True, so don't need to assign this to a variable name
#new_earnings.rename(columns = {0: 'earnings_imputed'}, inplace = True)
new_earnings.columns = ['earnings_imputed']
print(new_earnings)



dataset_missing = pd.concat([dataset_missing, new_earnings], axis = 1)
print(dataset_missing)

dataset_missing['earnings_missing'] = dataset_missing['earnings'].isnull()
print(dataset_missing)


### filling the missing value
dataset_missing['earnings'].fillna(dataset_missing['earnings_imputed'],inplace = True)

print(dataset_missing.isnull().sum())



### Drop columns are not necessary
dataset_missing = dataset_missing.drop(columns = ['earnings_imputed', 'earnings_missing'])




print(dataset_missing)


##dataset_missing.to_csv('data_not_missing.csv', index = False)
















