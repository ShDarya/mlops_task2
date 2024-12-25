import warnings

import pandas as pd
from model.train import train_model
from preprocessing.prepare_data import prepare_data
from utils.utils import plot_feature_importance, validate_model

warnings.filterwarnings('ignore')

import os
import sys

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# data config
data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# params
RANDOM_STATE = 42
TEST_SIZE = 0.3

# model path to save
model_path = '/Users/kshurik/Desktop/private/codes/mlops-course/lecture_2/artefacts/catboost_model.pkl'

if __name__ == '__main__':
    iris_data = pd.read_csv(data_path, header=None, names=col_names)
    X_train, X_test, y_train, y_test = prepare_data(iris_data, TEST_SIZE, RANDOM_STATE)
    model = train_model(X_train, y_train)
    validate_model(model, X_test, y_test)
    plot_feature_importance(model)
