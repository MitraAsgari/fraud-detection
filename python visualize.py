import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
from train import train_models
from preprocess import load_and_preprocess_data

def visualize_results(y_test, y_pred_xgb):
    # Visualize ROC Curve
    fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_clf.predict_proba(X_test)[:, 1])
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_clf.predict_proba(X_test)[:, 1])

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_log_reg, tpr_log_reg, label='Logistic Regression')
    plt.plot(fpr_rf, tpr_rf, label='Random Forest')
    plt.plot(fpr_xgb, tpr_xgb, label='XGBoost')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Display the Confusion Matrix for the best model
    conf_matrix = confusion_matrix(y_test, y_pred_xgb)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    log_reg, rf_clf, xgb_clf, y_pred_xgb, y_test = train_models(X_train, X_test, y_train, y_test)
    visualize_results(y_test, y_pred_xgb)
