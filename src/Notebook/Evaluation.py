import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
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


num_data = Data_Numeric[['Distance_km','Preparation_Time_min']]
cat_data =Data_categorie[['Weather','Traffic_Level']]
def use_scled(data):
    le = StandardScaler()
    scaled = le.fit_transform(data)
    scaled_numerique = pd.DataFrame(scaled, columns=data.columns)
    return scaled_numerique
def use_hote(data):
    la = OneHotEncoder(sparse_output=False)
    encoded = la.fit_transform(data)
    cooder_categorie = pd.DataFrame(encoded,columns = la.get_feature_names_out(data.columns))
    return cooder_categorie
cooder_categorie = use_hote(cat_data)
scaled_numerique = use_scled(num_data)

target = df['Delivery_Time_min']
prepared_data = pd.concat(
    [scaled_numerique, cooder_categorie, target],
    axis=1
)
X = df.drop(columns=["Delivery_Time_min"])
Y = df['Delivery_Time_min']
def split_data(df):
    X = df.drop(columns=["Delivery_Time_min"])
    Y = df['Delivery_Time_min']
    train_X , test_X , train_Y , test_Y = train_test_split(
    X , Y , test_size=0.2, random_state=42
    )
    return train_X,test_X,train_Y,test_Y

train_X,test_X,train_Y,test_Y = split_data(prepared_data)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_num),
        ('cat', OneHotEncoder(handle_unknown = 'ignore'), X_cat)
    ])