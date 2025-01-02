
# 0. Configuration

data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
RANDOM_STATE = 42
TEST_SIZE = 0.3
model_name = "CatBoost_Model"

# 1. Modules and functions
import warnings
import optuna
import mlflow
import pandas as pd
from model.train import train_model, call_opt
from preprocessing.prepare_data import prepare_data
from utils.utils import plot_feature_importance, validate_model, plot_shap_summary

warnings.filterwarnings('ignore')
# 2. Main
if __name__ == '__main__':

    # 2.1. Data preparation
    iris_data = pd.read_csv(data_path, header=None, names=col_names)
    X_train, X_test, y_train, y_test, X_tr, X_val, y_tr, y_val = prepare_data(iris_data, TEST_SIZE, RANDOM_STATE)

    with mlflow.start_run():

        # 2.2. Model training
        best_params, fig_opt = call_opt(X_tr, y_tr, X_val, y_val, n_trials=5, timeout=600)
        model = train_model(X_train, y_train, best_params)
        
    
        # Log best parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Save and register model
        mlflow.catboost.log_model(model, "model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, model_name)

        # 2.3. Model evaluation
        accuracy = validate_model(model, X_test, y_test)
        mlflow.log_metric("accuracy_score", accuracy)
        mlflow.log_metric("best_score", model.best_score_['learn']['MultiClass'])

        fig = plot_feature_importance(model)
        fig_shap = plot_shap_summary(X_train, model)

        # Log artifacts
        print(fig)
        print(fig_shap)
        mlflow.log_figure(fig, "feature_importance.png")
        mlflow.log_figure(fig_shap, "feature_shap.png")

    print(f"Model registered in MLFlow Registry as {model_name}")
