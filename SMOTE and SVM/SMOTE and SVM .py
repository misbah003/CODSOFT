import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import string
import os

nltk.download('punkt')
nltk.download('stopwords')

# -------------------------
# Helper Functions
# -------------------------
def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

def load_glove_embeddings(glove_file_path):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def get_average_vector(tokens, embeddings, dim=100):
    vectors = [embeddings[word] for word in tokens if word in embeddings]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)

# -------------------------
# Load and Process Data
# -------------------------
print("[INFO] Loading data...")

train_file = "C:/Users/HomeLaptop/Desktop/test/train_data.txt"
test_file = "C:/Users/HomeLaptop/Desktop/test/test_data.txt"
solution_file = "C:/Users/HomeLaptop/Desktop/test/test_data_solution.txt"
glove_file = "glove.6B.100d.txt"  # Must be in the same directory

train_df = pd.read_csv(train_file, sep=' ::: ', engine='python', names=['id', 'title', 'genre', 'plot'])
test_df = pd.read_csv(test_file, sep=' ::: ', engine='python', names=['id', 'title', 'plot'])
solution_df = pd.read_csv(solution_file, sep=' ::: ', engine='python', names=['id', 'genre'])

print("[INFO] Cleaning text...")
train_df['tokens'] = train_df['plot'].apply(clean_text)
test_df['tokens'] = test_df['plot'].apply(clean_text)

print("[INFO] Loading GloVe word embeddings...")
glove_embeddings = load_glove_embeddings(glove_file)

print("[INFO] Vectorizing text...")
X_train = np.vstack(train_df['tokens'].apply(lambda tokens: get_average_vector(tokens, glove_embeddings)))
X_test = np.vstack(test_df['tokens'].apply(lambda tokens: get_average_vector(tokens, glove_embeddings)))

# -------------------------
# Prepare Multi-label Target
# -------------------------
print("[INFO] Binarizing genres...")
y_train = train_df['genre'].apply(lambda g: g.split(', '))
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train)

# -------------------------
# Train One-vs-Rest SVM with SMOTE
# -------------------------
print("[INFO] Running SMOTE + SVM for each genre (One-vs-Rest)...")
y_preds = []

for i in range(y_train_bin.shape[1]):
    genre_name = mlb.classes_[i]
    print(f"  -> Training for genre: {genre_name}")
    y_col = y_train_bin[:, i]

    try:
        smote = SMOTE()
        X_res, y_res = smote.fit_resample(X_train, y_col)
    except ValueError:
        print(f"     [SKIPPED] Not enough samples for SMOTE for genre: {genre_name}")
        y_preds.append(np.zeros(len(X_test)))
        continue

    clf = SVC(kernel='linear')
    clf.fit(X_res, y_res)
    y_pred = clf.predict(X_test)
    y_preds.append(y_pred)

# -------------------------
# Final Prediction and Save
# -------------------------
print("[INFO] Generating final predictions...")
y_pred_final = np.array(y_preds).T  # shape: (num_samples, num_genres)
predicted_genres = mlb.inverse_transform(y_pred_final)

output_file = "predicted_genres.txt"
with open(output_file, "w", encoding="utf8") as f:
    for idx, genres in zip(test_df['id'], predicted_genres):
        genre_str = ", ".join(genres) if genres else "none"
        f.write(f"{idx} ::: {genre_str}\n")

print(f"✅ Predicted genres saved to: {output_file}")

# Optional: Evaluation
y_true = solution_df['genre'].apply(lambda g: g.split(', '))
y_true_bin = mlb.transform(y_true)
accuracy = accuracy_score(y_true_bin, y_pred_final)
print(f"\n🎯 Multi-label Genre Prediction Accuracy: {accuracy:.4f}")
