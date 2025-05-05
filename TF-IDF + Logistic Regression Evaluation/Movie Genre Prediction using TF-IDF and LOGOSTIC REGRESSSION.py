import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

# Load data
train_df = pd.read_csv('train_data.txt', sep=' ::: ', engine='python', names=['id', 'title', 'genre', 'plot'])
test_df = pd.read_csv('test_data.txt', sep=' ::: ', engine='python', names=['id', 'title', 'plot'])
test_solutions = pd.read_csv('test_data_solution.txt', sep=' ::: ', engine='python', names=['id', 'title', 'genre', 'plot'])

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

train_df['clean_plot'] = train_df['plot'].apply(clean_text)
test_df['clean_plot'] = test_df['plot'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['clean_plot'])
y_train = train_df['genre']

# Handle class imbalance with RandomOverSampler
ros = RandomOverSampler()
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_resampled, y_train_resampled)

# Prepare test data
test_df['id'] = test_df['id'].astype(str).str.strip()
test_solutions['id'] = test_solutions['id'].astype(str).str.strip()
test_solutions = test_solutions[['id', 'genre']]

# Merge test data with ground truth genres
test_merged = test_df.merge(test_solutions, on='id')
print(f"✅ Merged test set size: {len(test_merged)}")

# Predict and evaluate
X_test = vectorizer.transform(test_merged['clean_plot'])
y_test = test_merged['genre']
y_pred = lr_model.predict(X_test)

# Print classification report
print("\n📊 TF-IDF + Logistic Regression Evaluation (after oversampling):")
print(classification_report(y_test, y_pred, zero_division=0))

# Show genre distribution
print("\n📈 Original Training Genre Distribution:")
print(train_df['genre'].value_counts())
