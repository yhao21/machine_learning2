import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import haversine_distances
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from math import radians
from matplotlib import pyplot as plt



'''
predict price range first, then predict actual price
'''


class CleanData():

    def __init__(self, df):
        self.df = df

    
    def drop_missing_values(self):
        '''
        check missing values, and delete these rows

        missing:
        name                                 16
        host_name                            21
        last_review                       10052
        reviews_per_month                 10052
        '''

        df_missing_count = self.df.isnull().sum()
        df_missing = self.df.isnull()
        ## find index of those rows with missing values, should in list
        df_indexing = df_missing[(df_missing.last_review == True) | (df_missing.reviews_per_month == True)\
                | (df_missing.name == True) | (df_missing.host_name == True)].index.tolist()
        #print(df_indexing)
        ## delete these rows
        self.df = self.df.drop(index = df_indexing)
        self.df = self.df[[
            'price','neighbourhood_group','latitude', 'longitude', 'room_type','minimum_nights',\
            'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count',\
            'availability_365']]

        self.make_clean()


    
    def make_clean(self):
        
        ## form 7 price groups
        self.df['price_range'] = np.select([
            (self.df['price'] < 100),
            (self.df['price'] < 200),
            (self.df['price'] < 300),
            (self.df['price'] < 400),
            (self.df['price'] < 500),
            (self.df['price'] < 800),
            (self.df['price'] > 800),
            ],[0,1,2,3,4,5,6])

        ### plot latitude and longitude, colored by price groups
        #x = self.df['latitude']
        #y = self.df['longitude']
        #price_color = self.df['price_range']
        #plt.scatter(x,y,c =price_color)
        #plt.savefig('price_location.png')

        #print(self.df.columns.tolist())


        #### use 2021/02/24 as base, compute how long (days) there's no new review.
        now_date = datetime(2021,2,24)
        self.df['Date'] = pd.to_datetime(self.df['last_review'])
        self.df['last_review_date_diff'] = -(self.df['Date'] - now_date).apply(lambda x: x.days)
        self.df = self.df.drop(['Date','last_review'], axis = 1)
        
        dummy_list = [
                'room_type',
                'neighbourhood_group',
                ]



        #self.dummy_list = [
        #        'room_type',
        #        'neighbourhood_group',
        #        'minimum_nights',
        #        'number_of_reviews',
        #        'calculated_host_listings_count',
        #        'availability_365']
        

        order = ['price','price_range',  'latitude', 'longitude',\
                'room_type', 'minimum_nights', 'number_of_reviews',\
                'reviews_per_month', 'calculated_host_listings_count',\
                'availability_365', 'neighbourhood_group', 'last_review_date_diff']
        self.df = self.df[order]
        self.df = self.make_dummies(dummy_list)

        
        self.df.to_csv('full_clean_dataset.csv', index = False)








    def make_dummies(self, dummy_list):

        for item in dummy_list:
            dummy_col = pd.get_dummies(self.df[item])
            self.df = pd.concat([self.df,dummy_col], axis = 1)
        newdf = self.df.drop(dummy_list, axis = 1)

        return newdf



    def model_selection(self,split_number,machine,model_name,confusion = 0, r2_sign = 0, acc_sign = 0):

        target = self.df.iloc[:, 0].values
        data = self.df.iloc[:,1:].values
    
        kfold_object = KFold(n_splits=split_number)
        kfold_object.get_n_splits(data)
    
    
        results = []

        for training_index,test_index in kfold_object.split(data):
            data_training,data_test = data[training_index],data[test_index]
            target_training,target_test =target[training_index],target[test_index]
            machine.fit(data_training,target_training)
            prediction = machine.predict(data_test)
            if confusion == 1:
                conf_matrix = metrics.confusion_matrix(target_test,prediction)
                for i in conf_matrix:
                    print(i)
    
            if acc_sign == 1:
                results.append(metrics.accuracy_score(target_test,prediction))
            if r2_sign == 1:
                results.append(metrics.r2_score(target_test,prediction))
        
        print(model_name, results)







if __name__ == '__main__':

    #===========
    ## step1: make clean dataset, with price and price range(not dummy)

    #df = pd.read_csv('AB_NYC_2019.csv')
    #CleanData(df).drop_missing_values()



    #===========
    ## step2: select model for price machine(data with price range dummy)

    #df = pd.read_csv('full_clean_dataset.csv')
    #df = CleanData(df).make_dummies(['price_range'])

    #machine = linear_model.LinearRegression()
    #CleanData(df).model_selection(4,machine,'linear',r2_sign = 1)
    '''
    result:linear [0.502478944728638, 0.5404731004593971, 0.6459142809679543, 0.6697080012391131]
    '''

    #machine = linear_model.Lasso(alpha = 0.001, normalize=True)
    #CleanData(df).model_selection(4,machine,'lasso',r2_sign = 1)
    '''
    result: lasso [0.5023062458401142, 0.5404972388602172, 0.6459522423365893, 0.6700360429637205]
    '''

    #machine = linear_model.Ridge(alpha = 0.001, normalize=True)
    #CleanData(df).model_selection(4,machine,'ridge',r2_sign = 1)
    '''
    result:ridge [0.5023568942683851, 0.5404858720767283, 0.6459522567396866, 0.6698815602010382]
    '''


    ## save trained machine

    #with open('price_model.pickle','wb') as f:
    #    pickle.dump(machine,f)



    
    #===========
    ## step3: train PriceRange machine

    #df = pd.read_csv('full_clean_dataset.csv').drop(['price'], axis = 1)

    #machine = RandomForestClassifier(criterion= 'gini',n_estimators=9, max_depth=10, n_jobs=-1)
    #CleanData(df).model_selection(4,machine,'RF', acc_sign = 1)
    '''
    result:RF [0.6567071914279827, 0.6892323544564657, 0.6907779495105616, 0.6501803194229778]
    '''
    
    #machine = linear_model.LogisticRegression(multi_class='multinomial',n_jobs=-1,max_iter=5000)
    #CleanData(df).model_selection(4,machine,'logit', acc_sign = 1)


    # save PriceRange machine

    #with open('PriceRange_machine.pickle','wb') as g:
    #    pickle.dump(machine,g)



    #===========
    ## step4: Test machines
    '''
    Use NYC clean data, and drop price and price_range col. Then predict
    price_range, merge it to main df and use this new df predict price level, and
    then verify r2_score
    '''
    ##~~~~## sample test
    df = pd.read_csv('full_clean_dataset.csv')
    col_name = df.columns.tolist()
    # remove price and price_range
    del col_name[:2]
    true_target = df.iloc[:5000,0].values
    data = df.iloc[:5000,2:].values
    
    
    with open('PriceRange_machine.pickle','rb') as f:
        PriceRange_machine = pickle.load(f)
    price_range = pd.DataFrame(PriceRange_machine.predict(data), columns = ['price_range'])
    rest_VR = pd.DataFrame(data, columns = col_name)
    data = pd.concat([price_range,rest_VR], axis = 1)
    # don't forget to convert price_range to dummies
    data = CleanData(data).make_dummies(['price_range']).iloc[:,:].values
    
    with open('price_model.pickle', 'rb') as g:
        Price_machine = pickle.load(g)
    
    price_prediction = Price_machine.predict(data)
    R2 = metrics.r2_score(true_target, price_prediction)
    print(R2)
     
     
     
    










    ##~~~~## full dataset test
    #df = pd.read_csv('full_clean_dataset.csv')
    #col_name = df.columns.tolist()
    ## remove price and price_range
    #del col_name[:2]
    #true_target = df.iloc[:,0].values
    #data = df.iloc[:,2:].values


    #with open('PriceRange_machine.pickle','rb') as f:
    #    PriceRange_machine = pickle.load(f)
    #price_range = pd.DataFrame(PriceRange_machine.predict(data), columns = ['price_range'])
    #rest_VR = df.drop(['price', 'price_range'], axis = 1)
    #data = pd.concat([price_range,rest_VR], axis = 1)
    ## don't forget to convert price_range to dummies
    #data = CleanData(data).make_dummies(['price_range']).iloc[:,:].values

    #with open('price_model.pickle', 'rb') as g:
    #    Price_machine = pickle.load(g)

    #price_prediction = Price_machine.predict(data)
    #R2 = metrics.r2_score(true_target, price_prediction)
    #print(R2)





