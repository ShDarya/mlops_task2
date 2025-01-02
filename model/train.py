from catboost import CatBoostClassifier
import optuna
from sklearn.metrics import accuracy_score
from optuna.visualization import plot_parallel_coordinate, plot_optimization_history
import matplotlib.pyplot as plt

# params
RANDOM_STATE = 42
TEST_SIZE = 0.3

def train_model(X, y, params):

    
    model = CatBoostClassifier(
        **params
    )

    model.fit(X, y)

    return model


def opt(trial, X_tr, y_tr, X_val, y_val):

    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "MultiClass",  # For binary classification (could also be 'CrossEntropy' for multi-class)
        "verbose": 0,
    }

    model = CatBoostClassifier(**params, random_seed=RANDOM_STATE)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

    preds = model.predict(X_val)
    
   
    accuracy = accuracy_score(y_val, preds)
    
    return 1 - accuracy  

def call_opt(X_tr, y_tr, X_val, y_val, n_trials=5, timeout=600):
    # Create the study object
    study = optuna.create_study(direction="minimize")
    
    
    def objective(trial):
        return opt(trial, X_tr, y_tr, X_val, y_val)

    # Run the optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    fig = plt.figure(figsize=(8, 4))

    fig = plot_optimization_history(study)
    fig.show()


    # Get the best hyperparameters
    best_params = study.best_params
    return best_params, fig
