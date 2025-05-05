import pandas as pd
import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

# 1. Load data
train_df = pd.read_csv('train_data.txt', sep=' ::: ', engine='python', names=['id', 'title', 'genre', 'plot'])
test_df = pd.read_csv('test_data.txt', sep=' ::: ', engine='python', names=['id', 'title', 'plot'])
test_solutions = pd.read_csv('test_data_solution.txt', sep=' ::: ', engine='python', names=['id', 'title', 'genre', 'plot'])

# 2. Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

train_df['clean_plot'] = train_df['plot'].apply(clean_text)
test_df['clean_plot'] = test_df['plot'].apply(clean_text)

# 3. Word Embedding using Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['clean_plot'])
y_train = train_df['genre']

# 4. Handle class imbalance with RandomOverSampler
ros = RandomOverSampler()
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# 5. Train Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_resampled, y_train_resampled)

# 6. Prepare test data
test_df['id'] = test_df['id'].astype(str).str.strip()
test_solutions['id'] = test_solutions['id'].astype(str).str.strip()
test_solutions = test_solutions[['id', 'genre']]

# Merge test plots with solutions using ID
test_merged = test_df.merge(test_solutions, on='id')
print("✅ Merged test set size:", len(test_merged))

# 7. Transform test plots and predict
X_test = vectorizer.transform(test_merged['clean_plot'])
y_test = test_merged['genre']
y_pred = nb_model.predict(X_test)

# 8. Evaluation
print("\n📊 Word Embedding + Naive Bayes Evaluation (after oversampling):")
print(classification_report(y_test, y_pred, zero_division=0))

# 9. Show training genre distribution
print("\n📈 Original Training Genre Distribution:")
print(train_df['genre'].value_counts())
