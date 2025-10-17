import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from Gridseach import GridSearch_CV

path = "/Users/lait-zet/Desktop/Work_local/Data/data_livre.csv"
df = pd.read_csv(path)

def handle_missing(df, columns):
    i = 0
    while i < len(columns):
        col = columns[i]
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        i += 1
    return df

convert_columns = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Courier_Experience_yrs']
df = handle_missing(df, convert_columns)

def is_numeric(df):
    return df.select_dtypes(include=['int64', 'float64'])

def is_categorie(df):
    return df.select_dtypes(exclude=['int64', 'float64'])

Data_Numeric = is_numeric(df)
Data_categorie = is_categorie(df)

num_data = Data_Numeric[['Distance_km', 'Preparation_Time_min']]
cat_data = Data_categorie[['Weather', 'Traffic_Level']]

def use_scaled(data):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    return pd.DataFrame(scaled, columns=data.columns)

def use_hot(data):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(data)
    return pd.DataFrame(encoded, columns=encoder.get_feature_names_out(data.columns))

scaled_numerique = use_scaled(num_data)
encoded_categorie = use_hot(cat_data)
target = df['Delivery_Time_min']
prepared_data = pd.concat([scaled_numerique, encoded_categorie, target], axis=1)

def split_data(df):
    X = df.drop(columns=["Delivery_Time_min"])
    Y = df['Delivery_Time_min']
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
    return train_X, test_X, train_Y, test_Y

train_X, test_X, train_Y, test_Y = split_data(prepared_data)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_data.columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_data.columns)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('SVR', SVR())
])

paramR = {
    'SVR__C': [0.1, 1, 10],
    'SVR__kernel': ['linear', 'rbf'],
    'SVR__gamma': ['scale', 'auto']
}

results = GridSearch_CV(train_X, test_X, train_Y, test_Y)
print("resultas finale :", results)
