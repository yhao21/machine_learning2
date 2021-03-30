import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


dataset = pd.read_csv('dataset_ridge.csv')

## add polynomial terms
for i in range(1,10):
    dataset['x1_' + str(i)] = dataset['x1'] ** i

## interaction terms
##dataset['x1']*dataset['x2']

#target = dataset.iloc[:,1].values
##data = dataset.iloc[:,4:6].values
#data = dataset.iloc[:,4:9].values

print(dataset)


plt.plot(dataset['x1'], dataset['y'], '.')
plt.savefig('theplot.png')


## you may see from the graph, cubic terms might fit better
## and linear will be a mess





#machine = linear_model.LinearRegression()
#machine.fit(data, target)


print('='*20)
print('Ridge')
print('='*20)
for power in range(1,7):
    print('power %d' % power)
    target = dataset.iloc[:,1].values
    data = dataset.iloc[:,4:(power+4)].values
    machine = linear_model.Ridge(alpha = 0.001, normalize=True)
    machine.fit(data, target)
    np.set_printoptions(suppress=True)
    coef_value = machine.coef_
    print(coef_value)


print('='*20)
print('Lasso')
print('='*20)
for power in range(1,7):
    print('power %d' % power)
    target = dataset.iloc[:,1].values
    data = dataset.iloc[:,4:(power+4)].values
    machine = linear_model.Lasso(alpha = 0.001, normalize=True)
    machine.fit(data, target)
    np.set_printoptions(suppress=True)
    coef_value = machine.coef_
    print(coef_value)



### alpha: SSR + alpha* (sum of square of coefs)
### SSR and sum of square of coefs are in different unit, so alpha is a weight
### if alpha = 0 we back to linear regression
### we normalize because your data can have different scale
#machine = linear_model.Ridge(alpha = 0.001, normalize=True)
#machine.fit(data, target)
#
#
#
#coef_value = machine.coef_
###[-8.36416483  0.86987982] linearRegression result
#np.set_printoptions(suppress=True)
#print(coef_value)
#
#'''
#Linear regression
#
#
#Ridge
#before normalize
#[ 35.47591301 -17.41245344   3.12125005  -0.21084984   0.0036588 ]
#after normalized
#[-7.12542468  0.13436526  0.0923669   0.00678373 -0.00115032]
#
#x1^
#
#
#
#
#
#
#'''













