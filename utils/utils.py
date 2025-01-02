import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import shap

def validate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series):

   
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, preds))

   
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    return accuracy

def plot_feature_importance(model: object):
    

   
    feature_importance = model.get_feature_importance()
    feature_names = model.feature_names_
    sorted_idx = np.argsort(feature_importance)

    fig = plt.figure(figsize=(8, 4))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.title('Catboost Feature Importance')
    plt.show()

    return fig

def plot_shap_summary(X: pd.DataFrame, model: object):
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
  
    fig_shap = plt.figure(figsize=(8, 4))

    shap.summary_plot(shap_values, X)
    
    return fig_shap
