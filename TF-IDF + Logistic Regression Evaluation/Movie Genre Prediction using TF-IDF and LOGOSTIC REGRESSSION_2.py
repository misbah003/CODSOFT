import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')

# ---------------------------
# Utility Functions
# ---------------------------

def clean_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in tokens if word.isalpha() and word not in stop_words])

# ---------------------------
# Load Data
# ---------------------------

print("[INFO] Loading data...")
train_df = pd.read_csv("train_data.txt", sep=" ::: ", header=None, names=["ID", "TITLE", "GENRE", "DESCRIPTION"], engine='python')
test_df = pd.read_csv("test_data.txt", sep=" ::: ", header=None, names=["ID", "TITLE", "DESCRIPTION"], engine='python')

# ---------------------------
# Clean Text
# ---------------------------

print("[INFO] Cleaning text...")
train_df["clean_desc"] = train_df["DESCRIPTION"].apply(clean_text)
test_df["clean_desc"] = test_df["DESCRIPTION"].apply(clean_text)

# ---------------------------
# Prepare Target Labels
# ---------------------------

print("[INFO] Binarizing genres...")
train_df["Genre_List"] = train_df["GENRE"].apply(lambda x: x.split(","))
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_df["Genre_List"])

# ---------------------------
# Feature Extraction
# ---------------------------

print("[INFO] Extracting features (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=50000)
X = vectorizer.fit_transform(train_df["clean_desc"])
X_test = vectorizer.transform(test_df["clean_desc"])

# ---------------------------
# Train Logistic Regression
# ---------------------------

print("[INFO] Training Logistic Regression...")
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X, y)

# ---------------------------
# Predict
# ---------------------------

print("[INFO] Predicting...")
y_pred = model.predict(X_test)

# ---------------------------
# Save Predictions
# ---------------------------

print("[INFO] Saving predictions...")
predicted_genres = mlb.inverse_transform(y_pred)

output_df = pd.DataFrame({
    "ID": test_df["ID"],
    "Predicted_Genres": [",".join(genres) if genres else "None" for genres in predicted_genres]
})
output_df.to_csv("predictions.txt", sep="\t", index=False, header=False)

print("✅ Predictions saved to 'predictions.txt'")
