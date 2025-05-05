import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 1. Load data
train_df = pd.read_csv('train_data.txt', sep=' ::: ', engine='python', names=['id', 'title', 'genre', 'plot'])
test_df = pd.read_csv('test_data.txt', sep=' ::: ', engine='python', names=['id', 'title', 'plot'])
test_solutions = pd.read_csv('test_data_solution.txt', sep=' ::: ', engine='python', names=['id', 'genre'])


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
X_test = vectorizer.transform(test_df['clean_plot'])
y_train = train_df['genre']

# 4. Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# 5. Predict and Evaluate
y_pred = nb_model.predict(X_test)
print("TF-IDF + Naive Bayes Evaluation:")
print(classification_report(test_solutions['genre'], y_pred))
