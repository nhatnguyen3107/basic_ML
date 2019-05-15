import pandas as pd 
import os
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def load_data(train_file, test_file): #load data from file name
    data_train = pd.read_excel(train_file)
    data_test = pd.read_excel(test_file)
    return data_train, data_test

def clean_data(train_data_frame, test_data_frame): # clean null and duplicated points
    # REMOVE NULL DATA POINTS
    null_counter = train_data_frame.isnull().sum(axis=0)
    # print(test_data_frame.isnull().sum(axis = 0))
    for column, count_null in null_counter.items():
        if count_null > 0.3*train_data_frame.shape[0]:
            train_data_frame.drop(column, axis=1, inplace = True)
            test_data_frame.drop(column, axis=1, inplace = True)
    train_data_frame.dropna(inplace = True)
    # REMOVE DUPLICATED POINTS
    train_data_frame.drop_duplicates(keep = 'first', inplace = True)
    train_data_frame.to_excel('clean_data/clean_data_train.xlsx')  
    test_data_frame.to_excel('clean_data/clean_test_set.xlsx')
    return train_data_frame, test_data_frame
    
def feature_engineering(train_data_frame, test_data_frame):
    # Date_of_Journey Column processing, pick month and day_of_week (has a little range)
    train_data_frame['Month_of_Journey'] = pd.to_datetime(train_data_frame['Date_of_Journey'], format='%d/%m/%Y').dt.month.astype(int)
    train_data_frame['Day_of_Week'] = pd.to_datetime(train_data_frame['Date_of_Journey'], format='%d/%m/%Y').dt.dayofweek.astype(int)
    train_data_frame.drop('Date_of_Journey', axis=1, inplace=True)
    test_data_frame['Month_of_Journey'] = pd.to_datetime(test_data_frame['Date_of_Journey'], format='%d/%m/%Y').dt.month.astype(int)
    test_data_frame['Day_of_Week'] = pd.to_datetime(test_data_frame['Date_of_Journey'], format='%d/%m/%Y').dt.dayofweek.astype(int)
    test_data_frame.drop('Date_of_Journey', axis=1, inplace=True)

    # Source and Destination processing, synchronize 2 near place or be just the same place
    # print(train_data_frame['Source'].value_counts())
    # print(train_data_frame['Destination'].value_counts()) # non-clear: New Delhi and Delhi
    # print(test_data_frame['Source'].value_counts())
    # print(test_data_frame['Destination'].value_counts())  # non-clear: New Delhi and Delhi
    train_data_frame.replace({'Destination':{'New Delhi':'Delhi'}}, inplace=True)
    test_data_frame.replace({'Destination':{'New Delhi':'Delhi'}}, inplace=True)
    # print(train_data_frame['Destination'].value_counts()) # recheck
    # print(test_data_frame['Destination'].value_counts())  # recheck

    # Dep_Time and Arrival_Time Column processing, attribute time to period of day
    depart_hour_train = list(pd.to_datetime(train_data_frame['Dep_Time']).dt.hour)
    arrival_hour_train = list(pd.to_datetime(train_data_frame['Arrival_Time']).dt.hour)
    period_of_day = ['Morning', 'Noon', 'Evening', 'Night'] # [3->9h, 9->15h, 15->21h, 21->3h ]
    train_data_frame['Dep_Day_Period'] = [period_of_day[(x+3)//6-1] for x in depart_hour_train]
    train_data_frame['Arrival_Day_Period'] = [period_of_day[(x+3)//6-1] for x in arrival_hour_train]
    train_data_frame.drop(['Dep_Time','Arrival_Time'], axis=1, inplace=True)
    
    depart_hour_test = list(pd.to_datetime(test_data_frame['Dep_Time']).dt.hour)
    arrival_hour_test = list(pd.to_datetime(test_data_frame['Arrival_Time']).dt.hour)
    period_of_day = ['Morning', 'Noon', 'Evening', 'Night'] # [3->9h, 9->15h, 15->21h, 21->3h ]
    test_data_frame['Dep_Day_Period'] = [period_of_day[(x+3)//6-1] for x in depart_hour_test]
    test_data_frame['Arrival_Day_Period'] = [period_of_day[(x+3)//6-1] for x in arrival_hour_test]
    test_data_frame.drop(['Dep_Time','Arrival_Time'], axis=1, inplace=True)
    
    # Duration Column processing, attribute to one unit (minutes)
    duration_train = list(train_data_frame['Duration'])
    for i in range(len(duration_train)):
        spl = duration_train[i].split()
        if len(spl) == 1 and spl[0][-1] == 'h':
            duration_train[i] = duration_train[i] + ' 0m'
        elif len(spl) == 1 and spl[0][-1] == 'm':
            duration_train[i] = '0h ' + duration_train[i]
    train_data_frame['Duration'] = [int(duration_train[i].split()[0][:-1])*60 + int(duration_train[i].split()[1][:-1]) for i in range(len(duration_train))]
    train_data_frame['Duration'] = train_data_frame['Duration'].astype(int)

    duration_test = list(test_data_frame['Duration'])
    for i in range(len(duration_test)):
        spl = duration_test[i].split()
        if len(spl) == 1 and spl[0][-1] == 'h':
            duration_test[i] = duration_test[i] + ' 0m'
        elif len(spl) == 1 and spl[0][-1] == 'm':
            duration_test[i] = '0h ' + duration_test[i]
    test_data_frame['Duration'] = [int(duration_test[i].split()[0][:-1])*60 + int(duration_test[i].split()[1][:-1]) for i in range(len(duration_test))]
    test_data_frame['Duration'] = test_data_frame['Duration'].astype(int)


    
    #Total_Stop Column processing
    # print(train_data_frame['Total_Stops'].value_counts())
    train_data_frame.replace({'Total_Stops':{'non-stop':0, '1 stop':1, '2 stops': 2, '3 stops': 3, '4 stops': 4}}, inplace=True)
    test_data_frame.replace({'Total_Stops':{'non-stop':0, '1 stop':1, '2 stops': 2, '3 stops': 3, '4 stops': 4}}, inplace=True)
    train_data_frame['Total_Stops'] = train_data_frame['Total_Stops'].astype(int)
    test_data_frame['Total_Stops'] = test_data_frame['Total_Stops'].astype(int)
    
    #Additional_Info Column processing
    # print(train_data_frame['Additional_Info'].value_counts())
    train_data_frame.replace({'Additional_Info':{'No Info': 'No info'}}, inplace=True)
    test_data_frame.replace({'Additional_Info':{'No Info': 'No info'}}, inplace=True)
    # print(train_data_frame['Additional_Info'].value_counts())

    # print(train_data_frame.head())
    train_data_frame.to_excel('raw_feature/train_data_feature.xlsx')
    test_data_frame.to_excel('raw_feature/test_data_feature.xlsx')
    return train_data_frame, test_data_frame

def feat_decoding_and_scaling(train_set_feat, test_set_feat):
    X_feat_train = train_set_feat.drop('Price', axis=1)
    y_target_train = np.log1p(train_set_feat['Price'])
    X_feat_predict = test_set_feat

    X_categorical_feat_train = X_feat_train.select_dtypes(exclude=['int'])
    X_numerical_feat_train = X_feat_train.select_dtypes(include=['int'])
    X_categorical_feat_predict = X_feat_predict.select_dtypes(exclude=['int'])
    X_numerical_feat_predict = X_feat_predict.select_dtypes(include=['int'])
    # print(X_categorical_feat_predict.columns)
    # print(X_numerical_feat_predict.columns)
    
    # ENCODING CATEGORICAL FEATURES
    label_encoder = LabelEncoder()
    X_categorical_feat_train = X_categorical_feat_train.apply(label_encoder.fit_transform, axis=1)
    # print(X_categorical_feat_train.head())
    X_categorical_feat_predict = X_categorical_feat_predict.apply(label_encoder.fit_transform, axis=1)
    # print(X_categorical_feat_predict.head())

    # SCALING NUMERICAL FEATURES with Standard Scaler
    standard_scaler = StandardScaler()
    X_train_scaled = standard_scaler.fit_transform(X_numerical_feat_train)
    X_predict_scaled = standard_scaler.fit_transform(X_numerical_feat_predict)
    X_numerical_feat_train = pd.DataFrame(data=X_train_scaled, columns=X_numerical_feat_train.columns)
    X_numerical_feat_predict = pd.DataFrame(data=X_predict_scaled, columns=X_numerical_feat_predict.columns)
    # print(X_numerical_feat_train.head())
    train_set_feat_after = pd.concat([X_categorical_feat_train, X_numerical_feat_train], axis = 1)
    train_set_feat_after['Price'] = y_target_train
    predict_set_feat_after = pd.concat([X_categorical_feat_predict, X_numerical_feat_predict], axis = 1)
    # print(X_feat_train.head())
    train_set_feat_after.to_excel('final_feature/train_feat_after_tranforming.xlsx')
    predict_set_feat_after.to_excel('final_feature/predict_feat_after_tranforning.xlsx')
    # print(train_set_feat_after.shape, predict_set_feat_after.shape)



if __name__ == "__main__":
    raw_train_data_frame, raw_test_data_frame = load_data('raw_data/Data_Train.xlsx', 'raw_data/Test_set.xlsx')
    print('Train Data Shape: ', raw_train_data_frame.shape)
    print('Test Data Shape: ', raw_test_data_frame.shape)
    clean_train_data_frame, clean_test_data_frame = clean_data(raw_train_data_frame.copy(), raw_test_data_frame.copy()) #stored in clean_data
    print('\nAfter cleaning')
    print('Train Data Shape: {}. Removed {} column(s) and {} record(s)'.format(clean_train_data_frame.shape, raw_train_data_frame.shape[1]- clean_train_data_frame.shape[1], raw_train_data_frame.shape[0]- clean_train_data_frame.shape[0]))
    print('Test Data Shape: {}. Removed {} column(s) and {} record(s)'.format(clean_test_data_frame.shape, raw_test_data_frame.shape[1]- clean_test_data_frame.shape[1], raw_test_data_frame.shape[0]- clean_test_data_frame.shape[0]))
    train_set_feat, test_set_feat = feature_engineering(clean_train_data_frame.copy(), clean_test_data_frame.copy()) # stored in raw_feature
    print('\nAfter engineering data')
    print('The Number of features: ',train_set_feat.shape[1])
    print('Include: ', train_set_feat.columns.tolist())
    feat_decoding_and_scaling(train_set_feat, test_set_feat) #stored in final_feature
    print('Preprocessing data is complete!')

