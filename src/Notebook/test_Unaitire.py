import pytest
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
    df = pd.concat([X, y.rename("Delivery_Time_min")], axis=1)
    trainX, testX, trainY, testY = split_data(df)
    assert len(trainX) == len(trainY)
    assert len(testX) == len(testY)
