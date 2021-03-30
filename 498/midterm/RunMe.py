import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import linear_model as lm
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


'''
['ID', 'Case Number', 'Date', 'Block', 'IUCR', 'Primary Type', 'Description', 'L
ocation Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward', 'Communi
ty Area', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Year', 'Updated On', 'Lat
itude', 'Longitude', 'Location']

Frequency
print(df['Year'].value_counts())
2002    486767
2001    485793
2003    475961
2004    469399
2005    453744
2006    448143
2007    437051
2008    427115
2009    392782
2010    370414
2011    351891
2012    336161
2013    307335
2014    275586
2016    269520
2017    268737
2018    268311
2015    264500
2019    260549
2020    209939
2021     21390
'''


class CleanData():

    def __init__(self, df, count_crime = True):
        '''
        models == RF ---> RandomForestClassifier
        models == SVM ---> SVM
        models == Logistic ---> Logistic
        '''

        self.df = df
        self.crime_type_checklist = None
        self.count_crime_signal = count_crime

    def drop_missing_values(self):
        '''
        This function drops all obs/rows with missing value(s)
    
        Location Description    3308
        Ward                      28
        Community Area             1
        X Coordinate            7659
        Y Coordinate            7659
        Latitude                7659
        Longitude               7659
        Location                7659
        '''
        
        df_missing = self.df.isnull()
        missing_count = df_missing.sum()
        missing_index = df_missing[
                (df_missing['Location Description'] == True) |\
                (df_missing['Ward'] == True) |\
                (df_missing['Community Area'] == True) |\
                (df_missing['X Coordinate'] == True) |\
                (df_missing['Y Coordinate'] == True) |\
                (df_missing['Latitude'] == True) |\
                (df_missing['Longitude'] == True) |\
                (df_missing['Location'] == True)].index.tolist()
        self.df = self.df.drop(index = missing_index)


        self.drop_columns()
    

    def drop_columns(self):
#['ID', 'Case Number', 'Date', 'Block', 'IUCR', 'Primary Type', 'Description', 'L
#ocation Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward', 'Communi
#ty Area', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Year', 'Updated On', 'Lat
#itude', 'Longitude', 'Location']
        self.df = self.df.drop(['ID','Case Number','Description',\
                'Location','Location Description','IUCR','X Coordinate',\
                'Y Coordinate','Updated On','Beat','Unnamed: 0',\
                'Unnamed: 0.1','Block','FBI Code'], axis = 1)


        # convert Ture and False to 1 and 0 for these two col
        self.df['Arrest'] = self.df['Arrest'] * 1
        self.df['Domestic'] = self.df['Domestic'] * 1

        self.convert_datetime()


    def convert_datetime(self):
        '''
        extract datetime details
        '''
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Hours'] = self.df['Date'].dt.hour
        self.df['Hour_Slot'] = np.select([
            (self.df['Hours'] < 4),
            (self.df['Hours'] < 8),
            (self.df['Hours'] < 12),
            (self.df['Hours'] < 16),
            (self.df['Hours'] < 20),
            (self.df['Hours'] < 24),
            ],[0,1,2,3,4,5])
        self.df = self.df.drop(['Hours'], axis = 1)
        self.df = self.df.drop(['Date'], axis = 1)


        self.dummy_list = ['District','Community Area', 'Year']

        if self.count_crime_signal == True:
            self.count_crime_in_districts()
        else:
            self.make_dummies()
            self.df.to_csv('CleanData_without_crime_count.csv', index = False)
        #print(self.df)
        #self.model_selection_process()



    def count_crime_in_districts(self):

        '''
        Count # of different types of crimie happend in diff districts for
        each obs.
        Example:
        Counts # of Theft-type in each district.
        Counts # of Battery-type in each district.
        Counts # of Assault-type in each district.
        ...
        Each append each type, as a column, to df, i.e., 
        obs   crime_type    District    #of_theft   #of_battery 
        0       theft           20          15              55
        1       battery         20          15              55
        2       cri_damage      17          30              3


        Clearly, obs0 and obs1 are in district 20, and # of theft in district 20 is 
        15, # of batter in district 20 is 55. So I would say that it is more
        possible that Primary type for obs0 and obs1 is battery (55>15)
        '''

        ## a list of all district 
        self.district_unique_values = pd.unique(self.df['District']).tolist()
        ## a list of all crime type
        self.crime_type_unique_values = pd.unique(self.df['Primary Type']).tolist()

        self.district_crime_pre_df = []

        for district_item in self.district_unique_values:
            crime_types = self.df[self.df['District'] == district_item]['Primary Type']
            ## list all crime types ['THEFT','BATTERY',...]
            crime_index = crime_types.value_counts().index.tolist()
            crime_types_count = crime_types.value_counts().tolist()
            ### pair type and counts[('CRIM SEXUAL ASSAULT', 1), ('BATTERY', 1)]
            type_value_pair = [i for i in zip(crime_index,crime_types_count)]

            self.construct_crime_count_df(type_value_pair, district_item)
            self.district_crime_pre_df.append(self.district_row)
        #=======
        ## self.district_crime_pre_df preview: it is a list of lists
        #    District  BATTERY  CRIMINAL TRESPASS  CRIMINAL DAMAGE  DECEPTIVE PRACTICE  ...  KIDNAPPING  OBSCENITY  OTHER NARCOTIC VIOLATION  
        #0       10.0      810                 76              380                 137  ...           3          2                         0 
        #1       18.0      483                115              237                 561  ...           0          0                         0 
        #=======

        self.form_df_crime_count()
        col_name = self.df.columns.values.tolist() + self.full_district_type.columns.values.tolist()


        ## merge crime type and self.df
        self.df = self.df.iloc[:,:].values
        crime_type_df = self.full_district_type.iloc[:,:].values
        self.df = pd.DataFrame(np.hstack((self.df, crime_type_df)), columns = col_name)

        ## list dummy variables you need
        self.make_dummies()
        self.df.to_csv('CleanData_with_crime_count.csv', index = False)
        #print(self.df)

        #self.model_selection_process()




    def form_df_crime_count(self):
        crime_count_df = self.df['District'].values.tolist()
        self.full_district_type = []
        
        for one_district in crime_count_df:
            one_district_indexing = self.district_unique_values.index(one_district)
            sub_row_append = self.district_crime_pre_df[one_district_indexing]
            self.full_district_type.append(sub_row_append)
        self.full_district_type = pd.DataFrame(self.full_district_type)
        #print(self.full_district_type)






    def construct_crime_count_df(self, type_value_pair, district_item):
        self.district_row = [0 for i in range(len(self.crime_type_unique_values))]
        for i in type_value_pair:
            ## get place index, value + 1 since self.district_row's first value is District
            crime_indexing = self.crime_type_unique_values.index(i[0])
            self.district_row[crime_indexing] += i[1]





    def make_dummies(self):
        for item in self.dummy_list:
            dummy_col = pd.get_dummies(self.df[item])
            self.df = pd.concat([self.df,dummy_col], axis = 1)
        self.df = self.df.drop(self.dummy_list, axis = 1)

    


    def model_selection_process(self,models):
        target = self.df['Primary Type'].astype('category')
        ## convert categories to numbers
        checklist_col = target.cat.categories.tolist()
        target = target.cat.rename_categories([i for i in range(len(target.cat.categories.tolist()))])
        content = target.cat.categories.values
        self.crime_type_checklist = pd.DataFrame(content.reshape(1,len(content)), columns = checklist_col)
        #print(self.crime_type_checklist)
        self.crime_type_checklist.to_csv('Primary_type_checklist.csv')

        self.target = np.array(target.values)
        self.data = self.df.iloc[:,1:].values




        if models == 'RF':
            self.run_random_forest()
        elif models == 'SVM':
            self.run_svm()
        elif models == 'logit':
            self.run_logit()

        return self.results


    def run_logit(self):

        for ker in ['ovr','multinomial']:
            machine = lm.LogisticRegression(multi_class=ker, n_jobs=-1, max_iter=10000)
            self.run_kfold(4,machine)
            print(ker,self.acc_list)



    def run_svm(self):


        for ker in ['rbf', 'poly', 'sigmoid']:
            machine = svm.SVC(kernel=ker, decision_function_shape='ovr', max_iter=15000)
            self.run_kfold(4,machine)
            print(ker,self.acc_list)



    def run_random_forest(self):

        self.results = []
        indexing = 0
        for n_trees in [5,10,15,20,30,40,50,60,70,80,120,160,200]:
            for n_depth in [5,10,20,30,40,50,60,70,80]:
        #for n_trees in [5,10]:
        #   for n_depth in [5,10,15]:
               machine = RandomForestClassifier(criterion='gini',max_depth=n_depth,n_estimators=n_trees,n_jobs=-1)
               self.run_kfold(4,machine)
               #print(ker, n_depth, n_trees, self.acc_list)
               regression_info = [indexing, n_trees, n_depth, self.acc_list]
               self.results.append(regression_info)
               print(regression_info)
               indexing += 1
        self.results_df = pd.DataFrame(self.results, columns = ['Index','Trees','Depth','Acc_rate'])
        self.results_df.to_csv('RF_grid_search_results.csv', index = False)


        



    def run_kfold(self,split_number,machine,confusion = 0):
    
        kfold_object = KFold(n_splits=split_number)
        kfold_object.get_n_splits(self.data)
    
    
        self.acc_list = []
    
        for training_index,test_index in kfold_object.split(self.data):
            data_training,data_test = self.data[training_index],self.data[test_index]
            target_training,target_test =self.target[training_index],self.target[test_index]
            machine.fit(data_training,target_training)
            prediction = machine.predict(data_test)
            if confusion == 1:
                conf_matrix = metrics.confusion_matrix(target_test,prediction)
                for i in conf_matrix:
                    print(i)
            
    
    
            self.acc_list.append(metrics.accuracy_score(target_test,prediction))



def plot_acc_trend(results_list, figure_name):
    r2_series_x1 = []
    r2_series_x2 = []
    r2_series_x3 = []
    r2_series_x4 = []

    for item in results_list:
        model_parameter = item[0]
        x1 = item[3][0]     # first r2_score
        x2 = item[3][1]     # second r2_score
        x3 = item[3][2]     # third r2_score
        x4 = item[3][3]     # forth r2_score
        r2_series_x1.append(x1)
        r2_series_x2.append(x2)
        r2_series_x3.append(x3)
        r2_series_x4.append(x4)
        #index_checklist.append([placement, model_parameter])

    horizontal_values = [i for i in range(1,len(r2_series_x1)+1)]
    plt.plot(horizontal_values, r2_series_x1, c = 'red', label = 'x1')
    plt.plot(horizontal_values, r2_series_x2, c = 'green', label = 'x2')
    plt.plot(horizontal_values, r2_series_x3, c = 'blue', label = 'x3')
    plt.plot(horizontal_values, r2_series_x4, c = 'yellow', label = 'x4')
    plt.legend(loc = 'lower right')
    plt.savefig(figure_name)

if __name__ == '__main__':

    #=================
    # Step 1 Create dataset, year 2018-2020

    #df = pd.read_csv('Crimes_-_2001_to_Present.csv')
    #new_df = df[(df['Year'] == 2018) | (df['Year'] == 2019) | (df['Year'] == 2020) ]
    #print(new_df)
    #new_df.to_csv('sample_data.csv')


    #=================
    # Step 2 Create subset of sample data

    #df = pd.read_csv('sample_data.csv')
    #sub_df = df.sample(frac = 0.1).reset_index(drop = True)
    #sub_df.to_csv('sub_sample.csv')




    #=================
    # Step 3 Plot crimes on map

    #df = pd.read_csv('sub_sample.csv')
    #crime_type_list = pd.unique(df['Primary Type']).tolist()
    #if not os.path.exists('figures'):
    #    os.mkdir('figures')
    #for one_type in crime_type_list:
    #    x = df[df['Primary Type'] == one_type]['Latitude']
    #    y = df[df['Primary Type'] == one_type]['Longitude']
    #    plt.plot(x,y,'.')
    #    plt.savefig(os.path.join('figures',one_type + '.png'))
    #    plt.clf()

    


    #=================
    # Step 4 Clean Data

    #df = pd.read_csv('sub_sample.csv')
    #CleanData(df, count_crime=True).drop_missing_values()
    #CleanData(df, count_crime=False).drop_missing_values()



    #=================
    # Step 5 Model Selection
    df = pd.read_csv('CleanData_without_crime_count.csv')
    print(len(df.columns.tolist()))
    #results=CleanData(df).model_selection_process('RF')
    #figure_name = 'acc_trend_RF_without_crime_count.png'
    #plot_acc_trend(results, figure_name)
    #results=CleanData(df).model_selection_process('logit')
    #results=CleanData(df).model_selection_process('SVM')


    #df = pd.read_csv('CleanData_with_crime_count.csv')
    #results = CleanData(df).model_selection_process('RF')
    #figure_name = 'acc_trend_RF_with_crime_count.png'
    #plot_acc_trend(results, figure_name)
    #results = CleanData(df).model_selection_process('logit')
    #results = CleanData(df).model_selection_process('SVM')









