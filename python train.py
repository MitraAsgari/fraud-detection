from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from preprocess import load_and_preprocess_data

def train_models(X_train, X_test, y_train, y_test):
    # Create and train Logistic Regression model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Create and train Random Forest model
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)

    # Create and train XGBoost model
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.fit(X_train, y_train)

    # Predictions with Logistic Regression
    y_pred_log_reg = log_reg.predict(X_test)
    y_prob_log_reg = log_reg.predict_proba(X_test)[:, 1]

    # Predictions with Random Forest
    y_pred_rf = rf_clf.predict(X_test)
    y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

    # Predictions with XGBoost
    y_pred_xgb = xgb_clf.predict(X_test)
    y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]

    # Calculate and print metrics
    print('Logistic Regression:')
    print(f'AUC-ROC: {roc_auc_score(y_test, y_prob_log_reg)}')
    print(f'Precision: {precision_score(y_test, y_pred_log_reg)}')
    print(f'Recall: {recall_score(y_test, y_pred_log_reg)}')

    print('Random Forest:')
    print(f'AUC-ROC: {roc_auc_score(y_test, y_prob_rf)}')
    print(f'Precision: {precision_score(y_test, y_pred_rf)}')
    print(f'Recall: {recall_score(y_test, y_pred_rf)}')

    print('XGBoost:')
    print(f'AUC-ROC: {roc_auc_score(y_test, y_prob_xgb)}')
    print(f'Precision: {precision_score(y_test, y_pred_xgb)}')
    print(f'Recall: {recall_score(y_test, y_pred_xgb)}')

    return log_reg, rf_clf, xgb_clf, y_pred_xgb, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    train_models(X_train, X_test, y_train, y_test)
