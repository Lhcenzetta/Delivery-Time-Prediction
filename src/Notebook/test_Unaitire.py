import pytest
from sklearn.metrics import mean_absolute_error
from Gridseach import GridSearch_CV  
from Evaluation import split_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
@pytest.fixture
def sample_data():
    data = {
        'age': [20, 30, 40, 50],
        'income': [2000, 3000, 4000, 5000],
        'churn': [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    X = df[['age', 'income']]
    y = df['churn']
    return X, y
def test_split_dimensions(sample_data):
    X, y = sample_data
    df = pd.concat([X, y], axis=1)
    df = df.rename(columns={'churn': 'Churn'})
    trainX, testX, trainY, testY = split_data(df)
    assert len(trainX) == len(trainY)
    assert len(testX) == len(testY)
X = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100)
})
y = X['feature1'] * 10 + X['feature2'] * 5 + np.random.randn(100)

train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)
def test_mae_threshold():
    """
    Vérifie que la MAE maximale des modèles ne dépasse pas le seuil défini.
    """
    results = GridSearch_CV(train_X, test_X, train_Y, test_Y)

    MAX_MAE = 10.0

    mae_RF = results["RandomForestRegressor"]["MAE"]
    mae_SVR = results["SVR"]["MAE"]

    assert mae_RF < MAX_MAE, f"MAE RandomForest trop élevée: {mae_RF:.2f}"
    assert mae_SVR < MAX_MAE, f"MAE SVR trop élevée: {mae_SVR:.2f}"
