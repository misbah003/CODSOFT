# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MultiLabelBinarizer
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load datasets
train_df = pd.read_csv('train_data.txt', delimiter=' ::: ', header=None, names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
test_df = pd.read_csv('test_data.txt', delimiter=' ::: ', header=None, names=['ID', 'TITLE', 'DESCRIPTION'])

# Data Preprocessing
train_df['DESCRIPTION'] = train_df['DESCRIPTION'].fillna('')
test_df['DESCRIPTION'] = test_df['DESCRIPTION'].fillna('')

# Convert 'GENRE' to list of genres
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_df['GENRE'].apply(lambda x: x.split(',')))

# TF-IDF Vectorization of 'DESCRIPTION' column
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train = tfidf_vectorizer.fit_transform(train_df['DESCRIPTION'])
X_test = tfidf_vectorizer.transform(test_df['DESCRIPTION'])

# Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define and train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_res, y_train_res)

# Define and train XGBoost model
xgb_model = XGBClassifier(scale_pos_weight=1, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_res, y_train_res)

# Predict with Logistic Regression
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_f1_micro = f1_score(y_test, lr_predictions, average='micro')
lr_f1_macro = f1_score(y_test, lr_predictions, average='macro')

# Predict with XGBoost
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_f1_micro = f1_score(y_test, xgb_predictions, average='micro')
xgb_f1_macro = f1_score(y_test, xgb_predictions, average='macro')

# Output comparison between Logistic Regression and XGBoost
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
print(f"Logistic Regression Micro F1: {lr_f1_micro:.4f}")
print(f"XGBoost Micro F1: {xgb_f1_micro:.4f}")
print(f"Logistic Regression Macro F1: {lr_f1_macro:.4f}")
print(f"XGBoost Macro F1: {xgb_f1_macro:.4f}")

# Hyperparameter Tuning for XGBoost
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_grid_search = GridSearchCV(estimator=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                               param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
xgb_grid_search.fit(X_train_res, y_train_res)

# Best parameters from GridSearchCV
print(f"Best parameters for XGBoost: {xgb_grid_search.best_params_}")

# Retrain XGBoost with the best parameters
best_xgb_model = xgb_grid_search.best_estimator_
best_xgb_model.fit(X_train_res, y_train_res)

# Re-predict with the best XGBoost model
best_xgb_predictions = best_xgb_model.predict(X_test)

# Output final evaluation metrics for best XGBoost
best_xgb_accuracy = accuracy_score(y_test, best_xgb_predictions)
best_xgb_f1_micro = f1_score(y_test, best_xgb_predictions, average='micro')
best_xgb_f1_macro = f1_score(y_test, best_xgb_predictions, average='macro')

print(f"Best XGBoost Accuracy: {best_xgb_accuracy:.4f}")
print(f"Best XGBoost Micro F1: {best_xgb_f1_micro:.4f}")
print(f"Best XGBoost Macro F1: {best_xgb_f1_macro:.4f}")
