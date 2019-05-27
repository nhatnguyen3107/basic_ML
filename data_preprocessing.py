import pandas as pd 
import os
import numpy as np 
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

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
    
def convert_time_to_day_period(data_frame, column):
    hour = list(pd.to_datetime(data_frame[column]).dt.hour)
    period_of_day = ['Morning', 'Noon', 'Evening', 'Night'] # [3->9h, 9->15h, 15->21h, 21->3h ]
    data_frame['Day_Period_of_' + column] = [period_of_day[(x+3)//6-1] for x in hour]
    return data_frame.drop(column, axis=1)

def merge_value(data_frame, column, val_source, val_des):
    return data_frame.replace({column:{'New Delhi':'Delhi'}})

def synchronize_format_time(data_frame, column):
    copy_list = list(data_frame[column])
    for i in range(len(copy_list)):
        _split = copy_list[i].split()
        if len(_split) == 1 and _split[0][-1] == 'h':
            copy_list[i] = copy_list[i] + ' 0m'
        elif len(_split) == 1 and _split[0][-1] == 'm':
            copy_list[i] = '0h ' + copy_list[i]
    data_frame[column] = copy_list
    return data_frame

def parse_hour_to_minute(data_frame, column):
    copy_list = list(data_frame[column])
    data_frame[column] = [int(copy_list[i].split()[0][:-1])*60 + int(copy_list[i].split()[1][:-1]) for i in range(len(copy_list))]
    data_frame[column] = data_frame[column].astype(float)
    return data_frame

def generate_features(data_frame):
    # Date_of_Journey Column processing, pick month and day_of_week (has a little range)
    data_frame['Month_of_Journey'] = pd.to_datetime(data_frame['Date_of_Journey'], format='%d/%m/%Y').dt.month.astype(float)
    data_frame['Day_of_Week'] = pd.to_datetime(data_frame['Date_of_Journey'], format='%d/%m/%Y').dt.dayofweek.astype(float)
    data_frame.drop('Date_of_Journey', axis=1, inplace=True)

    # Source and Destination processing, synchronize 2 near place or be just the same place
    data_frame = merge_value(data_frame, 'Destination', 'New Delhi', 'Delhi')

    # Dep_Time and Arrival_Time Column processing, attribute time to period of day
    data_frame = convert_time_to_day_period(data_frame, 'Dep_Time')
    data_frame = convert_time_to_day_period(data_frame, 'Arrival_Time')

    # Duration Column processing, attribute to one unit (minutes)
    data_frame = synchronize_format_time(data_frame, 'Duration')
    data_frame = parse_hour_to_minute(data_frame, 'Duration')
    
    #Total_Stop Column processing
    data_frame.replace({'Total_Stops':{'non-stop':0, '1 stop':1, '2 stops': 2, '3 stops': 3, '4 stops': 4}}, inplace=True)
    data_frame['Total_Stops'] = data_frame['Total_Stops'].astype(float)
    
    #Additional_Info Column processing
    data_frame = merge_value(data_frame, 'Additional_Info', 'No Info', 'No info')

    # print(data_frame.head())
    return data_frame

def feat_encoding_and_scaling(X_feats_set, encode=True, scale=True):
    X_categorical_feats = X_feats_set.select_dtypes(exclude=['int', 'float'])
    X_numerical_feats = X_feats_set.select_dtypes(include=['int', 'float'])
    # ENCODING CATEGORICAL FEATURES
    if encode == True:
        X_categorical_feats = encode_feat_with_label_encoder(X_categorical_feats)

    # SCALING NUMERICAL FEATURES with Standard Scaler
    if scale == True:
        X_numerical_feats = scale_feat_with_standard_scaler(X_numerical_feats)

    X_categorical_feats.reset_index(drop=True, inplace=True)
    X_numerical_feats.reset_index(drop=True, inplace=True)
    X_feats_set_after = pd.concat([X_categorical_feats, X_numerical_feats], axis = 1)
    return X_feats_set_after

def encode_feat_with_label_encoder(X_categorical_feats):
    label_encoder = LabelEncoder()
    return X_categorical_feats.apply(label_encoder.fit_transform)

def scale_feat_with_standard_scaler(X_numerical_feats):
    standard_scaler = StandardScaler()
    X_scaled = standard_scaler.fit_transform(X_numerical_feats)
    return pd.DataFrame(data=X_scaled, columns=X_numerical_feats.columns)

def evaluate_result(y_to_test, y_to_predict):
    print('Accuracy: ', metrics.r2_score(y_to_test, y_to_predict))
    print('Mean Square Error: ', metrics.mean_squared_error(y_to_test, y_to_predict))

def select_features_by_random_forest_regressor(X, y):
    X_to_train, X_to_test, y_to_train, y_to_test = train_test_split(X, y, random_state=42, test_size=0.3)
    model = RandomForestRegressor(n_estimators=20, max_depth=5,random_state=42)
    model.fit(X_to_train, y_to_train)
    y_pred = model.predict(X_to_test)
    feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['Importance rate']).sort_values('Importance rate', ascending=False)
    chosen_columns = feature_importances.loc[feature_importances['Importance rate']>=0.01]
    return chosen_columns.index.tolist()


def reduce_features_by_pca(X):
    pca = PCA(n_components='mle')
    fit = pca.fit_transform(X)
    # new_X = pca.transform(fit)
    new_data = pd.DataFrame(data=fit)
    # print('Explained_variance: ', fit.explained_variance_)
    # print('Ratio: ', fit.explained_variance_ratio_)
    # print('Components: ', fit.components_)
    return new_data

def predict_by_SVR(data_frame):
    X = data_frame.values[:,:-1]
    y = data_frame.values[:,-1]
    X_to_train, X_to_test, y_to_train, y_to_test = train_test_split(X, y, random_state=42, test_size=0.3)
    svr = SVR(kernel= 'rbf', degree=4, gamma='auto')
    svr.fit(X_to_train, y_to_train)
    y_to_predict = svr.predict(X_to_test)
    evaluate_result(y_to_test, y_to_predict)

def predict_by_linear_regression(data_frame):
    X = data_frame.values[:,:-1]
    y = data_frame.values[:,-1]
    X_to_train, X_to_test, y_to_train, y_to_test = train_test_split(X, y, random_state=42, test_size=0.3)
    lireg = LinearRegression(fit_intercept=True, normalize=True)
    lireg.fit(X_to_train, y_to_train)
    y_to_predict = lireg.predict(X_to_test)
    evaluate_result(y_to_test, y_to_predict)

def predict_by_RFR_combine_randomsearchCV(data_frame):
    X = data_frame.values[:,:-1]
    y = data_frame.values[:,-1]
    X_to_train, X_to_test, y_to_train, y_to_test = train_test_split(X, y, random_state=42, test_size=0.3)
    estimator = RandomForestRegressor()
    cv = ShuffleSplit(n_splits=5, random_state=42, test_size=0.3)
    random_grid = {'max_features' : ['auto', 'sqrt', 'log2'],
                    'n_estimators': [10, 18, 22, 200, 700],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf' : [1, 2, 4]}
    rf_random = RandomizedSearchCV(estimator=estimator, param_distributions=random_grid, cv=cv, random_state=42, n_jobs=4)
    rf_random.fit(X_to_train, y_to_train)
    y_to_predict = rf_random.predict(X_to_test)
    print('Best hyper-para after tuning', rf_random.best_params_)
    evaluate_result(y_to_test, y_to_predict)

def predict_non_label_data_by_RFR(train_data, predict_data):
    X_to_train = train_data.drop('Price', axis=1)
    y_to_train = train_data['Price']
    rfr = RandomForestRegressor(n_estimators=700, min_samples_split=2, min_samples_leaf=2, max_features = 'auto')
    rfr.fit(X_to_train, y_to_train)
    y_predict = rfr.predict(predict_data)
    return y_predict


if __name__ == "__main__":
    raw_train_data_frame, raw_test_data_frame = load_data('raw_data/Data_Train.xlsx', 'raw_data/Test_set.xlsx')
    print('Train Data Shape: ', raw_train_data_frame.shape)
    print('Test Data Shape: ', raw_test_data_frame.shape)
    
    clean_train_data_frame = clean_data(raw_train_data_frame.copy())
    clean_test_data_frame = raw_test_data_frame.copy()
    clean_train_data_frame.to_excel('clean_data/clean_data_train.xlsx', index=False)  
    clean_test_data_frame.to_excel('clean_data/clean_test_set.xlsx', index=False)
    print('\nAfter cleaning')
    print('Train Data Shape: {}. Removed {} column(s) and {} record(s)'.format(clean_train_data_frame.shape, raw_train_data_frame.shape[1]- clean_train_data_frame.shape[1], raw_train_data_frame.shape[0]- clean_train_data_frame.shape[0]))
    print('Test Data Shape: {}. Removed {} column(s) and {} record(s)'.format(clean_test_data_frame.shape, raw_test_data_frame.shape[1]- clean_test_data_frame.shape[1], raw_test_data_frame.shape[0]- clean_test_data_frame.shape[0]))
    
    train_feat_frame = generate_features(clean_train_data_frame.copy())
    test_feat_frame =  generate_features(clean_test_data_frame.copy())
    train_feat_frame.to_excel('raw_feature/train_data_feature.xlsx', index=False)
    test_feat_frame.to_excel('raw_feature/test_data_feature.xlsx',index=False)
    print('\nAfter engineering data')
    print('The Number of features: ',train_feat_frame.shape[1])
    print('Include: ', train_feat_frame.columns.tolist())
    
    X_feats_train = train_feat_frame.drop('Price', axis=1)
    y_target_train = np.log1p(train_feat_frame['Price'])
    # print(y_target_train.values)
    # final_train_data = feat_encoding_and_scaling(X_feats_train)
    tranformed_train_data = feat_encoding_and_scaling(X_feats_train)
    tranformed_train_data['Price'] = y_target_train.values
    tranformed_test_data = feat_encoding_and_scaling(test_feat_frame)
    # tranformed_test_data = feat_encoding_and_scaling(test_feat_frame, scale=False)
    tranformed_train_data.to_excel('tranformed_feature/train_data_after_tranforming.xlsx',index=False)
    tranformed_test_data.to_excel('tranformed_feature/test_data_after_tranforning.xlsx',index=False)
    
    # selected_features = select_features_by_random_forest_regressor(tranformed_train_data.drop('Price', axis=1),y_target_train.values)
    # selected_train_data = tranformed_train_data.filter(items= selected_features + ['Price'], axis=1)
    # selected_test_data = tranformed_test_data.filter(items= selected_features)
    # print('\nAfter selecting feature \nRemaining features: ', selected_features)
    selected_train_data = reduce_features_by_pca(tranformed_train_data.drop('Price', axis=1).values)
    selected_train_data['Price'] = y_target_train.values
    selected_test_data = reduce_features_by_pca(tranformed_test_data.values)
    selected_train_data.to_excel('final_feature/train_data_after_selecting.xlsx', index=False)
    selected_test_data.to_excel('final_feature/test_data_after_selecting.xlsx', index=False)

    print('\nTraining data...')
    # predict_by_SVR(selected_train_data)
    # predict_by_linear_regression(selected_train_data)
    # predict_by_RFR_combine_randomsearchCV(selected_train_data)
    y_target_predict = predict_non_label_data_by_RFR(selected_train_data, selected_test_data)
    predicted_result = pd.DataFrame(data=np.exp(y_target_predict).astype(int), index=None, columns=['Price'])
    predicted_result.to_excel('result.xlsx', index=False)
    print('Preprocessing data is complete!')

