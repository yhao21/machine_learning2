import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras import layers
import keras
import  numpy as np



def run_kfold(split_number,data,target,machine,confusion = 0):

    kfold_object = KFold(n_splits=split_number)
    kfold_object.get_n_splits(data)
    result = []
    for training_index,test_index in kfold_object.split(data):
        data_training,data_test = data[training_index],data[test_index]
        target_training,target_test =target[training_index],target[test_index]
        machine.fit(data_training,target_training)
        prediction = machine.predict(data_test)
        result.append(metrics.r2_score(target_test,prediction))


    return result






df = pd.read_csv('audio_extraction_dataset.csv')


df = df.drop(['filename', 'Unnamed: 0'], axis = 1)
print(df.head())


target = df.iloc[:,-1].values
encoder = LabelEncoder()
target = encoder.fit_transform(target)

data = df.iloc[:,:-1].values
scaler = StandardScaler()
data = scaler.fit_transform(data)

print(target)


#machine = RandomForestClassifier(n_estimators=20, max_depth=20, n_jobs=-1)


#data_training, data_test, target_training, target_test = train_test_split(data, target, test_size=)



kfold_object = KFold(n_splits=4)
kfold_object.get_n_splits(data)
result = []
for training_index,test_index in kfold_object.split(data):
    data_training,data_test = data[training_index],data[test_index]
    target_training,target_test =target[training_index],target[test_index]


    machine = Sequential()
    ## more layers make it more accurate
    machine.add(layers.Dense(256,activation='relu',input_shape = (data_training.shape[1],)))
    machine.add(layers.Dense(128, activation='relu'))
    machine.add(layers.Dense(64, activation='relu'))
    ## at end add how many songs you have
    machine.add(layers.Dense(80, activation='softmax'))
    machine.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
    machine.fit(data_training, target_training, epochs = 100, batch_size=128)



    prediction = np.argmax(machine.predict(data_test), axis = -1)
    result.append(metrics.accuracy_score(target_test, prediction))
    #result.append(metrics.r2_score(target_test,prediction))

#results = run_kfold(4, data, target, machine)
print(result)

