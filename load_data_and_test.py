import pandas as pd 
import os
import numpy as np 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from data_preprocessing import load_data
from data_preprocessing import evaluate_result

def predict_by_SVR(X, y):
    X_to_train, X_to_test, y_to_train, y_to_test = train_test_split(X, y, random_state=42, test_size=0.3)
    svr = SVR(kernel= 'rbf', degree=4, gamma='auto')
    svr.fit(X_to_train, y_to_train)
    y_to_predict = svr.predict(X_to_test)
    evaluate_result(y_to_test, y_to_predict)

def predict_by_linear_regression(X, y):
    X_to_train, X_to_test, y_to_train, y_to_test = train_test_split(X, y, random_state=42, test_size=0.3)
    linear = LinearRegression(fit_intercept=True, normalize=True)
    linear.fit(X_to_train, y_to_train)
    y_to_predict = linear.predict(X_to_test)
    evaluate_result(y_to_test, y_to_predict)

def predict_by_logistic_regression(X, y): #error
    X_to_train, X_to_test, y_to_train, y_to_test = train_test_split(X, y, random_state=42, test_size=0.3)
    logic = LogisticRegression(solver='newton-cg', multi_class='auto')
    logic.fit(X_to_train, y_to_train.astype(int))
    y_to_predict = logic.predict(X_to_test)
    print(y_to_predict)
    evaluate_result(y_to_test, y_to_predict)

def predict_by_RFR_combine_randomsearchCV(X, y):
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

def predict_by_neural_network(X, y):
    X_to_train, X_to_test, y_to_train, y_to_test = train_test_split(X, y, random_state=42, test_size=0.3)
    neural = MLPRegressor(hidden_layer_sizes=200, max_iter=200)
    neural.fit(X_to_train, y_to_train)
    y_to_predict = neural.predict(X_to_test)
    evaluate_result(y_to_test, y_to_predict)

if __name__ == "__main__":
    training_data, predicting_data = load_data('final_feature/train_data_after_selecting.xlsx', 'final_feature/test_data_after_selecting.xlsx')
    X = training_data.values[:,:-1]
    y = training_data.values[:,-1]

    # predict_by_linear_regression(X, y)        #accuracy = 0.5292941437111072
    # predict_by_SVR(X, y)                      #accuracy=0.8513313784153758
    # predict_by_RFR_combine_randomsearchCV(X, y) #accuracy=0.8524250566810908
    # predict_by_logistic_regression(X, y) #non_useful, maybe only apply to classification problem
    # predict_by_neural_network(X, y)           #accuracy = 0.7916816635000207
    # excute_neural_network_model(X, y, X.shape[1])
