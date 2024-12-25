import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(data: pd.DataFrame, test_size: float, random_state: int = 42):

    # encode the target variable
    data['class'] = data['class'].astype('category').cat.codes

    # split data into features (X) and target (y)
    X = data.drop(columns=['class'])
    y = data['class']

    X_with_features = calculate_feautres(X)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_with_features, y, test_size=test_size, random_state=random_state
    )

    print(f'Train size is: {X_train.shape} \nTest size is: {X_test.shape}')

    return X_train, X_test, y_train, y_test


def calculate_feautres(data: pd.DataFrame):

    # let's do some basic feature generation -- initial feature to the power of 2
    data['petal_length_squared'] = data['petal_length'] ** 2
    data['petal_width_squared'] = data['petal_width'] ** 2

    return data
