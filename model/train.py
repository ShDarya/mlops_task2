from catboost import CatBoostClassifier


def train_model(X, y):
    # fit the model -- no need for too high values for hyperparams for the scarce data
    model = CatBoostClassifier(
        iterations=100, learning_rate=0.1, depth=3, verbose=False
    )

    model.fit(X, y)

    return model
