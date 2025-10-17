
from sklearn.svm import SVR
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from Evaluation import train_X,test_X, train_Y,test_Y


def GridSearch_CV(trainX, testX ,trainY ,testY):
    RF = RandomForestRegressor()
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 1.0]
    }
    grid_search_for_RF = GridSearchCV(
        estimator = RF,
        param_grid = param_grid,
        scoring='r2',
        cv = 5,
        n_jobs=-1,
        verbose=1
    )
    grid_search_for_RF.fit(trainX,trainY)
    best_RF = grid_search_for_RF.best_estimator_



    svr = SVR()
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    grid_search_for_SVC = GridSearchCV(
        estimator = svr,
        param_grid = param_grid,
        cv = 5,
        scoring='r2',
        verbose= 0
    )
    grid_search_for_SVC.fit(trainX,trainY)
    best_SVR = grid_search_for_SVC.best_estimator_



    pred_RF = best_RF.predict(testX)
    pred_SVR = best_SVR.predict(testX)

    mae_RF = mean_absolute_error(testY, pred_RF)
    r2_RF = r2_score(testY, pred_RF)

    mae_SVR = mean_absolute_error(testY, pred_SVR)
    r2_SVR = r2_score(testY, pred_SVR)

    print(f"Random Forest - MAE: {mae_RF:.3f}, R²: {r2_RF:.3f}")
    print(f"SVR            - MAE: {mae_SVR : .3f}, R²: {r2_SVR:.3f}")

    if r2_RF > r2_SVR:
        print("\n final choisi est: Random Forest Regressor")
    else:
        print("\nfinal choisi est: SVR (Support Vector Regressor)")
    return {
        "RandomForestRegressor" : {"MAE":mae_RF, "r_2" : r2_RF},
        "SVR" : {"MAE":mae_RF, "r_2" : r2_SVR}
    }
result = GridSearch_CV(train_X,test_X, train_Y,test_Y)
print(result)
