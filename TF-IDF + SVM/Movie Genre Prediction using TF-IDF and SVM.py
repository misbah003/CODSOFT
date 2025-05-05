import pandas as pd
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

print("✅ Script started (with SMOTE and class weights).")

# Load data
train_df = pd.read_csv('train_data.txt', sep=' ::: ', engine='python', names=['id', 'title', 'genre', 'plot'])
test_df = pd.read_csv('test_data.txt', sep=' ::: ', engine='python', names=['id', 'title', 'plot'])
test_solutions = pd.read_csv('test_data_solution.txt', sep=' ::: ', engine='python', names=['id', 'title', 'genre', 'plot'])

print("✅ Data loaded.")
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

train_df['clean_plot'] = train_df['plot'].apply(clean_text)
test_df['clean_plot'] = test_df['plot'].apply(clean_text)

print("✅ Text cleaned.")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X_train = vectorizer.fit_transform(train_df['clean_plot'])
y_train = train_df['genre']
print(f"✅ TF-IDF features extracted. Shape of X_train: {X_train.shape}")

# SMOTE Oversampling
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"✅ Oversampling completed. Shape of X_train_resampled: {X_train_resampled.shape}")
print(f"✅ Oversampling completed. Number of unique classes in y_train_resampled: {len(set(y_train_resampled))}")

# Train LinearSVC
start_time = time.time()
svm_model = LinearSVC(class_weight='balanced', max_iter=5000, verbose=1)
svm_model.fit(X_train_resampled, y_train_resampled)
print(f"✅ LinearSVC model trained. Time taken: {round(time.time() - start_time, 2)} seconds.")

# Prepare test data
test_df['id'] = test_df['id'].astype(str).str.strip()
test_solutions['id'] = test_solutions['id'].astype(str).str.strip()
test_merged = test_df.merge(test_solutions[['id', 'genre']], on='id')
print(f"✅ Merged test set size: {len(test_merged)}")

# Predict
X_test = vectorizer.transform(test_merged['clean_plot'])
y_test = test_merged['genre']
y_pred = svm_model.predict(X_test)

# Evaluation
print("\n📊 TF-IDF + LinearSVC Evaluation (Optimized):")
print(classification_report(y_test, y_pred, zero_division=0))

# Genre distribution
print("\n📈 Training Genre Distribution (Original):")
print(train_df['genre'].value_counts())
