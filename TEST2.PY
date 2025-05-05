import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

# 1. Load data
train_df = pd.read_csv('train_data.txt', sep=' ::: ', engine='python', names=['id', 'title', 'genre', 'plot'])
test_df = pd.read_csv('test_data.txt', sep=' ::: ', engine='python', names=['id', 'title', 'plot'])
test_solutions = pd.read_csv('test_data_solution.txt', sep=' ::: ', engine='python', names=['id', 'title', 'genre', 'plot'])

# 2. Clean plot text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

train_df['clean_plot'] = train_df['plot'].apply(clean_text)
test_df['clean_plot'] = test_df['plot'].apply(clean_text)

# 3. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['clean_plot'])
y_train = train_df['genre']

# 4. Handle class imbalance with oversampling
ros = RandomOverSampler()
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# 5. Train Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_resampled, y_train_resampled)

# 6. Clean and match IDs
test_df['id'] = test_df['id'].astype(str).str.strip()
test_solutions['id'] = test_solutions['id'].astype(str).str.strip()

# Ensure we're using the correct columns: `id` and `genre` only
test_solutions = test_solutions[['id', 'genre']]  # Only take id and genre

# Debugging print
print("\nSample Test IDs:", test_df['id'].head().tolist())
print("Sample Solution IDs:", test_solutions['id'].head().tolist())

# Merge test data with ground truth genres
test_merged = test_df.merge(test_solutions, on='id')
print(f"Merged test set size: {len(test_merged)}")

# Safety check
if len(test_merged) == 0:
    print("❌ ERROR: No matching IDs found between test data and test solutions.")
    exit()

# 7. Predict and Evaluate
X_test_filtered = vectorizer.transform(test_merged['clean_plot'])
y_pred = nb_model.predict(X_test_filtered)

print("\nTF-IDF + Naive Bayes Evaluation (after oversampling):")
print(classification_report(test_merged['genre'], y_pred, zero_division=0))

# 8. Show genre distribution
print("\nOriginal Training Genre Distribution:")
print(train_df['genre'].value_counts())
