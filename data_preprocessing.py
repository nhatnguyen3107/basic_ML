import pandas as pd 
import os
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def load_data(train_file, test_file): #load data from file name
    data_train = pd.read_excel(train_file)
    data_test = pd.read_excel(test_file)
    return data_train, data_test

def clean_data(data_frame): # clean null and duplicated points
    # REMOVE NULL DATA POINTS
    null_counter = data_frame.isnull().sum(axis=0)
    for column, count_null in null_counter.items():
        if count_null > 0.3*data_frame.shape[0]:
            data_frame.drop(column, axis=1, inplace = True)
    data_frame.dropna(inplace = True)
    # REMOVE DUPLICATED POINTS
    data_frame.drop_duplicates(keep = 'first', inplace = True)
    return data_frame
    
def feature_engineering(data_frame):
    # Date_of_Journey Column processing, pick month and day_of_week (has a little range)
    data_frame['Month_of_Journey'] = pd.to_datetime(data_frame['Date_of_Journey'], format='%d/%m/%Y').dt.month.astype(float)
    data_frame['Day_of_Week'] = pd.to_datetime(data_frame['Date_of_Journey'], format='%d/%m/%Y').dt.dayofweek.astype(float)
    data_frame.drop('Date_of_Journey', axis=1, inplace=True)

    # Source and Destination processing, synchronize 2 near place or be just the same place
    # print(data_frame['Source'].value_counts())
    # print(data_frame['Destination'].value_counts()) # non-clear: New Delhi and Delhi
    data_frame.replace({'Destination':{'New Delhi':'Delhi'}}, inplace=True)
    # print(data_frame['Destination'].value_counts()) # recheck

    # Dep_Time and Arrival_Time Column processing, attribute time to period of day
    depart_hour_train = list(pd.to_datetime(data_frame['Dep_Time']).dt.hour)
    arrival_hour_train = list(pd.to_datetime(data_frame['Arrival_Time']).dt.hour)
    period_of_day = ['Morning', 'Noon', 'Evening', 'Night'] # [3->9h, 9->15h, 15->21h, 21->3h ]
    data_frame['Dep_Day_Period'] = [period_of_day[(x+3)//6-1] for x in depart_hour_train]
    data_frame['Arrival_Day_Period'] = [period_of_day[(x+3)//6-1] for x in arrival_hour_train]
    data_frame.drop(['Dep_Time','Arrival_Time'], axis=1, inplace=True)
    
    # Duration Column processing, attribute to one unit (minutes)
    duration_train = list(data_frame['Duration'])
    for i in range(len(duration_train)):
        spl = duration_train[i].split()
        if len(spl) == 1 and spl[0][-1] == 'h':
            duration_train[i] = duration_train[i] + ' 0m'
        elif len(spl) == 1 and spl[0][-1] == 'm':
            duration_train[i] = '0h ' + duration_train[i]
    data_frame['Duration'] = [int(duration_train[i].split()[0][:-1])*60 + int(duration_train[i].split()[1][:-1]) for i in range(len(duration_train))]
    data_frame['Duration'] = data_frame['Duration'].astype(float)
    
    #Total_Stop Column processing
    # print(data_frame['Total_Stops'].value_counts())
    data_frame.replace({'Total_Stops':{'non-stop':0, '1 stop':1, '2 stops': 2, '3 stops': 3, '4 stops': 4}}, inplace=True)
    data_frame['Total_Stops'] = data_frame['Total_Stops'].astype(float)
    
    #Additional_Info Column processing
    # print(data_frame['Additional_Info'].value_counts())
    data_frame.replace({'Additional_Info':{'No Info': 'No info'}}, inplace=True)
    # print(data_frame['Additional_Info'].value_counts())

    # print(data_frame.head())
    return data_frame

def feat_decoding_and_scaling(X_feats_set):
    X_categorical_feats = X_feats_set.select_dtypes(exclude=['int', 'float'])
    X_numerical_feats = X_feats_set.select_dtypes(include=['int', 'float'])
    # ENCODING CATEGORICAL FEATURES
    label_encoder = LabelEncoder()
    X_categorical_feats = X_categorical_feats.apply(label_encoder.fit_transform)

    # SCALING NUMERICAL FEATURES with Standard Scaler
    standard_scaler = StandardScaler()
    X_scaled = standard_scaler.fit_transform(X_numerical_feats)
    X_numerical_feats = pd.DataFrame(data=X_scaled, columns=X_numerical_feats.columns)
    X_categorical_feats.reset_index(drop=True, inplace=True)
    X_numerical_feats.reset_index(drop=True, inplace=True)
    X_feats_set_after = pd.concat([X_categorical_feats, X_numerical_feats], axis = 1)
    return X_feats_set_after

def select_features_by_random_forest_regressor(X, y):
    X_to_train, X_to_test, y_to_train, y_to_test = train_test_split(X, y, random_state=42, test_size=0.3)
    model = RandomForestRegressor(n_estimators=20, max_depth=5,random_state=42)
    model.fit(X_to_train, y_to_train)
    y_pred = model.predict(X_to_test)
    feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['Importance rate']).sort_values('Importance rate', ascending=False)
    chosen_columns = feature_importances.loc[feature_importances['Importance rate']>=0.02]
    print(chosen_columns)

if __name__ == "__main__":
    raw_train_data_frame, raw_test_data_frame = load_data('raw_data/Data_Train.xlsx', 'raw_data/Test_set.xlsx')
    print('Train Data Shape: ', raw_train_data_frame.shape)
    print('Test Data Shape: ', raw_test_data_frame.shape)
    
    clean_train_data_frame = clean_data(raw_train_data_frame.copy())
    clean_test_data_frame = clean_data(raw_test_data_frame.copy())
    clean_train_data_frame.to_excel('clean_data/clean_data_train.xlsx')  
    clean_test_data_frame.to_excel('clean_data/clean_test_set.xlsx')
    print('\nAfter cleaning')
    print('Train Data Shape: {}. Removed {} column(s) and {} record(s)'.format(clean_train_data_frame.shape, raw_train_data_frame.shape[1]- clean_train_data_frame.shape[1], raw_train_data_frame.shape[0]- clean_train_data_frame.shape[0]))
    print('Test Data Shape: {}. Removed {} column(s) and {} record(s)'.format(clean_test_data_frame.shape, raw_test_data_frame.shape[1]- clean_test_data_frame.shape[1], raw_test_data_frame.shape[0]- clean_test_data_frame.shape[0]))
    
    train_feat_frame = feature_engineering(clean_train_data_frame.copy())
    test_feat_frame =  feature_engineering(clean_test_data_frame.copy())
    train_feat_frame.to_excel('raw_feature/train_data_feature.xlsx', index=False)
    test_feat_frame.to_excel('raw_feature/test_data_feature.xlsx',index=False)
    print('\nAfter engineering data')
    print('The Number of features: ',train_feat_frame.shape[1])
    print('Include: ', train_feat_frame.columns.tolist())
    
    X_feats_train = train_feat_frame.drop('Price', axis=1)
    y_target_train = np.log1p(train_feat_frame['Price'])
    final_train_data = feat_decoding_and_scaling(X_feats_train)
    final_train_data['Price'] = y_target_train
    final_test_data = feat_decoding_and_scaling(test_feat_frame)
    final_train_data.to_excel('final_feature/train_feat_after_tranforming.xlsx',index=False)
    final_test_data.to_excel('final_feature/predict_feat_after_tranforning.xlsx',index=False)
    
    select_features_by_random_forest_regressor(final_train_data.drop('Price', axis=1),y_target_train)
    print('Preprocessing data is complete!')

