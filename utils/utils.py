import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def validate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series):

    # check the performance
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, preds))


def plot_feature_importance(model: object):
    """
    model MUST have feature_names_ & feature_importances_ properties
    (boosting like algos have it)
    """

    # let's look at the importance of the features
    feature_importance = model.get_feature_importance()
    feature_names = model.feature_names_
    sorted_idx = np.argsort(feature_importance)

    fig = plt.figure(figsize=(8, 4))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.title('Catboost feature importance')
    plt.show()
