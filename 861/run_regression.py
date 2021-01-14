import pandas as pd





dataset = pd.read_csv('dataset_3_outputs-1.csv')
print(dataset)


target = dataset.iloc[:,0].values
print(target)
