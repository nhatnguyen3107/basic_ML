import pandas as pd 
import os
import numpy as np 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from data_preprocessing import load_data
from data_preprocessing import evaluate_result
import tensorflow as tf

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

def neural_network_model(X, dimension):
    weight_0 = tf.Variable(tf.random_uniform([dimension, 10]))
    bias_0 = tf.Variable(tf.zeros([10]))
    layer_0 = tf.add(tf.matmul(X, weight_0), bias_0)
    layer_0 = tf.nn.relu(layer_0)

    weight_1 = tf.Variable(tf.random_uniform([10, 10]))
    bias_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(layer_0, weight_1), bias_1)
    layer_1 = tf.nn.relu(layer_1)
    
    weight_2 = tf.Variable(tf.random_uniform([10, 1]))
    bias_2 = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_1, weight_2), bias_2)

    return output

def excute_neural_network_model(X, y, dimension):
    X_to_train, X_to_test, y_to_train, y_to_test = train_test_split(X, y, random_state=42, test_size=0.3)
    xs = tf.placeholder('float')
    ys = tf.placeholder('float')
    output = neural_network_model(xs, dimension)
    cost = tf.reduce_mean(tf.square(output - ys))
    train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2):
            for j in range(X_to_train.shape[0]):
                sess.run([cost, train], feed_dict = {xs: X_to_train[j,:].reshape(1,dimension), ys:y_to_train[j]})
        y_pred = sess.run(output, feed_dict={xs:X_to_test})
    evaluate_result(y_to_test, y_pred)


if __name__ == "__main__":
    training_data, predicting_data = load_data('final_feature/train_data_after_selecting.xlsx', 'final_feature/test_data_after_selecting.xlsx')
    X = training_data.values[:,:-1]
    y = training_data.values[:,-1]
    # predict_by_RFR_combine_randomsearchCV(X, y)
    # predict_by_logistic_regression(X, y)
    excute_neural_network_model(X, y, X.shape[1])
