import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

# Load data
print("✅ Loading data...")
train_df = pd.read_csv('train_data.txt', sep=' ::: ', engine='python', names=['id', 'title', 'genre', 'plot'])
test_df = pd.read_csv('test_data.txt', sep=' ::: ', engine='python', names=['id', 'title', 'plot'])
test_solutions = pd.read_csv('test_data_solution.txt', sep=' ::: ', engine='python', names=['id', 'title', 'genre', 'plot'])

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

print("✅ Cleaning text...")
train_df['clean_plot'] = train_df['plot'].apply(clean_text)
test_df['clean_plot'] = test_df['plot'].apply(clean_text)

# Load GloVe embeddings
print("✅ Loading GloVe embeddings...")
glove_path = r"C:\Users\HomeLaptop\Desktop\test\New folder\glove.6B.300d.txt"
embeddings_index = {}
with open(glove_path, encoding='utf8') as f:
    for line in tqdm(f, desc="Reading GloVe vectors"):
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vector

embedding_dim = 300

# Average Word Embedding Vectorizer
def get_avg_embedding(text, embeddings_index, dim):
    words = text.split()
    valid_vectors = [embeddings_index[word] for word in words if word in embeddings_index]
    if valid_vectors:
        return np.mean(valid_vectors, axis=0)
    else:
        return np.zeros(dim)

# Create embeddings for training and testing data
print("✅ Creating embedding vectors...")
X_train = np.vstack(train_df['clean_plot'].apply(lambda x: get_avg_embedding(x, embeddings_index, embedding_dim)))
y_train = train_df['genre']

X_test = np.vstack(test_df['clean_plot'].apply(lambda x: get_avg_embedding(x, embeddings_index, embedding_dim)))

# Handle class imbalance
ros = RandomOverSampler()
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Train logistic regression
print("✅ Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_resampled, y_train_resampled)

# Merge test data with ground truth
test_df['id'] = test_df['id'].astype(str).str.strip()
test_solutions['id'] = test_solutions['id'].astype(str).str.strip()
test_merged = test_df.merge(test_solutions[['id', 'genre']], on='id')
y_test = test_merged['genre']

# Predict and evaluate
print("✅ Predicting and evaluating...")
y_pred = lr_model.predict(X_test)

print("\n📊 Word Embedding + Logistic Regression Evaluation:")
print(classification_report(y_test, y_pred, zero_division=0))

# Show training distribution
print("\n📈 Training Genre Distribution:")
print(train_df['genre'].value_counts())
