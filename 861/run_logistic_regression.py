
import pandas as pd
from sklearn import linear_model







dataset = pd.read_csv('dataset_3_outputs-1.csv')
print(dataset)


target = dataset.iloc[:,1].values
data = dataset.iloc[:,3:9].values



machine = linear_model.LinearRegression()
machine = linear_model.LogisticRegression()
machine.fit(data, target)


new_data = [
    [0.01, -0.2, 0.5, 1.1, 0, 0],
    [-0.5, -0.1, 0.44, 0.9, 1, 0.5]
]

new_target = machine.predict(new_data)
print(new_target)



