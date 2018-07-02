import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import decomposition ,ensemble, metrics ,neighbors
from sklearn.svm import SVR
from scipy import *
import random
import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model

num_of_splits = 1
num=6


def mean(list):
    list.fillna(list.mean())
    return list


def convert_date(list,X):
    month_list = []
    hour_list = []
    week_list=[]
    for li in list:
        d=datetime.datetime.strptime(li,"%Y-%m-%d %H:%M:%S")
        month_list.append(d.month)
        hour_list.append(d.hour)
        week_list.append(d.weekday())
    X = np.insert(X, 0, month_list, axis=1)
    X = np.insert(X, 1, hour_list, axis=1)
    X = np.insert(X, 2, week_list, axis=1)
    return X


def centralize(training_x):
    columns = [training_x[:, 0], training_x[:, 1], training_x[:, 2], training_x[:, 3], training_x[:, 4],
               training_x[:, 5], training_x[:, 6], training_x[:, 7],training_x[:, 8], training_x[:, 9]]

    changing_values = [np.mean(training_x[:, 0]), np.mean(training_x[:, 1]), np.mean(training_x[:, 2]),
                       np.mean(training_x[:, 3]), np.mean(training_x[:, 4]), np.mean(training_x[:, 5]),
                       np.mean(training_x[:, 6]), np.mean(training_x[:, 7]), np.mean(training_x[:, 8]), np.mean(training_x[:, 9])]

    for i in range(len(training_x[0])):
        for j in range(len(columns[0])):
            training_x[:, i][j] -= changing_values[i]
    return training_x


def normalize(training_x):
    columns = [training_x[:, 0], training_x[:, 1], training_x[:, 2], training_x[:, 3], training_x[:, 4],
               training_x[:, 5], training_x[:, 6], training_x[:, 7], training_x[:, 8], training_x[:, 9]]

    changing_values = [np.std(training_x[:, 0]), np.std(training_x[:, 1]), np.std(training_x[:, 2]),
                       np.std(training_x[:, 3]), np.std(training_x[:, 4]), np.std(training_x[:, 5]),
                       np.std(training_x[:, 6]), np.std(training_x[:, 7]), np.std(training_x[:, 8]), np.std(training_x[:, 9])]

    for i in range(len(training_x[0])):
        for j in range(len(columns[0])):
            if changing_values[i]!=0:
                training_x[:, i][j] /= changing_values[i]

    return training_x


def rmse(test_y, test_y_prediction):
    difference = np.subtract(test_y, test_y_prediction)
    square = np.multiply(difference, difference)
    mean_squared_error = float(np.mean(square))
    return np.sqrt(mean_squared_error)


def cv_optimize(clf, parameters, X, y, n_jobs=1, n_folds=5, score_func=None):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
    else:
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds)
    gs.fit(X, y)
    print("BEST", gs.best_params_, gs.best_score_, gs.grid_scores_)
    best = gs.best_estimator_
    return best


def main():
    training_set = pd.read_csv(sys.argv[1], low_memory=False)
    training_set = shuffle(training_set)
    training_x_date = np.array(training_set.get("pickup_datetime"))
    training_x = np.array(training_set.iloc[:, [3, 7, 9, 10, 11, 12, 13]].values)
    training_x = convert_date(training_x_date, training_x)
    training_y = np.array(training_set.get("trip_time_in_secs"))
    training_x_new = []
    training_y_new = []

    for i in range(len(training_x)):
        isNotNan = True
        for clan in training_x[i]:
            if np.isnan(clan) or clan == 0:
                isNotNan = False
                break
        if isNotNan:
            if training_x[i][5] != 0 and not np.isnan(training_x[i][5]):
                rand = random.randrange(150)
                if rand >= 146 and 700 > training_y[i] / training_x[i][5] > 125:
                    training_x_new.append(training_x[i])
                    training_y_new.append(training_y[i])
    training_x = normalize(centralize(np.array(training_x_new)))
    training_y = np.array(training_y_new)

    pca = decomposition.PCA(n_components=10)
    pca.fit(training_x)
    training_x = pca.transform(training_x)

    test_set = pd.read_csv(sys.argv[2])
    test_x_date = np.array(test_set.get("pickup_datetime"))
    test_x = np.array(test_set.iloc[:, [3, 7, 9, 10, 11, 12, 13]].values)
    test_x = convert_date(test_x_date, test_x)
    test_y = np.array(test_set.get("trip_time_in_secs"))
    test_x_new=[]
    test_y_new=[]
    for i in range(len(test_x)):
        isNotNan=True
        for clan in test_x[i]:
            if np.isnan(clan):
                isNotNan=False
                break
        if isNotNan:
            rand=random.randrange(150)
            if rand>=148:
                test_x_new.append(test_x[i])
                test_y_new.append(test_y[i])
    test_y = np.array(test_y_new)
    test_x = normalize(centralize(np.array(test_x_new)))
    test_x = pca.transform(test_x)
    print("odradio")

    # print("duzina x:", len(training_x))
    # print("duzina test_x:", len(test_x))
    # print("srednja vrednost y:", np.mean(test_y))

    # ************ K-Nearest Neighbors Regressor ********
    knn = neighbors.KNeighborsRegressor()
    # knn_parameters = {"n_neighbors": [1, 2, 5, 10, 20]}
    # knn_best = cv_optimize(knn, knn_parameters, training_x, training_y, score_func='neg_mean_squared_error')
    knn.fit(test_x, test_y)
    predict3 = knn.predict(test_x)
    rmse3 = rmse(test_y, predict3)
    print('*' * 30)
    print("K-Nearest Neighbors Regressor")
    print("RMSE: ", rmse3)
    print("r2_score test data: ", metrics.r2_score(test_y, predict3))
    print('*' * 30)
    # ***************************************************

"""
    # **************** MLP Regressor ********************
    mlpr = MLPRegressor(hidden_layer_sizes=(300,), solver="adam", batch_size=300, learning_rate="adaptive",
                        max_iter=1000)
    # mlpr_parameters = {"hidden_layer_sizes":[(150,)],"solver":["adam"],"batch_size":[300],"learning_rate":['adaptive']}
    # mlpr_best = cv_optimize(mlpr, mlpr_parameters, training_x, training_y, score_func='neg_mean_squared_error')
    mlpr.fit(training_x, training_y)

    predict = mlpr.predict(test_x)
    print('*'*30)
    print("MLP Regressor")
    print("RMSE test data: ", rmse(test_y, predict))
    print(test_y, predict)
    print("r2_score training data:", mlpr.score(training_x, training_y))
    print("r2_score test data:", mlpr.score(test_x, test_y))
    print('*' * 30)

    # ***************************************************

    # **************** Bagging Regressor ****************
    br = ensemble.BaggingRegressor(n_estimators=100, max_samples=1.0, max_features=1.0)
    br.fit(training_x, training_y)
    predictBR = br.predict(test_x)
    print('*' * 30)
    print("Bagging Regressor")
    print("RMSE: ", rmse(test_y, predictBR))
    print("r2_score test data: ", metrics.r2_score(test_y, predictBR))
    print('*' * 30)
    # ***************************************************

    # **********Gradient Boosting Regressor**************
    feature_labels = np.array(['month', 'hour', 'week', 'rate_code', 'passenger_count', 'trip_distance'
                                  , "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"])
    est = ensemble.GradientBoostingRegressor(n_estimators=10, learning_rate=0.0001, max_depth=10, random_state=0, loss='ls')
    est.fit(training_x, training_y)
    predict2 = est.predict(test_x)
    rmse2 = rmse(test_y, predict2)
    print('*' * 30)
    print("Gradient Booosting Regressor")
    print("RMSE: ", rmse2)
    print("r2_score test data: ", metrics.r2_score(test_y, predict2))
    print('*' * 30)
    # print("prosek:",test_y.mean())
    # print(predict2-test_y)
    # print(rmse1)
    importance = est.feature_importances_
    feature_indexes_by_importance = importance.argsort()
    for index in feature_indexes_by_importance:
        print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] * 100.0)))

    # ***************************************************

    

    # ************ Ada Boost Regressor ******************
    ada = ensemble.AdaBoostRegressor(n_estimators=50, learning_rate=0.001, loss='square')
    ada.fit(test_x, test_y)
    predict4 = ada.predict(test_x)
    rmse4 = rmse(test_y, predict4)
    print('*' * 30)
    print("Ada Boost Regressor")
    print("RMSE: ", rmse4)
    print("r2_score test data: ", metrics.r2_score(test_y, predict4))
    print('*' * 30)
    # ***************************************************

    # ************ Lasso Regressor **********************
    lasso = linear_model.Lasso(alpha=0.01, max_iter=1000, tol=0.0001)
    lasso.fit(test_x, test_y)
    predict5 = lasso.predict(test_x)
    rmse5 = rmse(test_y, predict5)
    print("r2_score test data: ", metrics.r2_score(test_y, predict5))
    print('*' * 30)
    print("Lasso Regressor")
    print("RMSE: ", rmse5)
    print('*' * 30)

    # ***************************************************

    # ************ Elastic Net Regressor ****************

    e_net = linear_model.ElasticNet(alpha=0.01, max_iter=1000, tol=0.0001)
    e_net.fit(test_x, test_y)
    predict6 = e_net.predict(test_x)
    rmse6 = rmse(test_y, predict6)
    print('*' * 30)
    print("Elastic Net Regressor")
    print("RMSE: ", rmse6)
    print("r2_score test data: ", metrics.r2_score(test_y, predict6))
    print('*' * 30)

    # ***************************************************
"""


if __name__ == '__main__':
    main()
