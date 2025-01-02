import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np



def prepare_data(data: pd.DataFrame, test_size: float, random_state: int = 42):

    
    data['class'] = data['class'].astype('category').cat.codes
    
    
    X = data.drop(columns=['class'])
    y = data['class']

    X_with_features = calculate_features(X)

    X_with_features = X_with_features.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_with_features), columns=X_with_features.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    print(f'Train size is: {X_train.shape} \nTest size is: {X_test.shape}')

    return X_train, X_test, y_train, y_test, X_tr, X_val, y_tr, y_val

def calculate_features(data: pd.DataFrame):
    
    data['petal_length_squared'] = data['petal_length'] ** 2
    data['petal_width_squared'] = data['petal_width'] ** 2

    data['petal_length_width_ratio'] = data['petal_length'] / (data['petal_width'] + 1e-9)
    data['sepal_length_width_ratio'] = data['sepal_length'] / (data['sepal_width'] + 1e-9)

    data['petal_length_x_width'] = data['petal_length'] * data['petal_width']
    data['sepal_length_x_width'] = data['sepal_length'] * data['sepal_width']

    data['log_petal_length'] = np.log1p(data['petal_length'])
    data['log_petal_width'] = np.log1p(data['petal_width'])

    return data
