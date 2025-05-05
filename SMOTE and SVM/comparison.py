import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from collections import Counter

# --- Load ground truth ---
true_ids = []
true_labels = []

with open("test_data_solution.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ::: ")
        if len(parts) >= 3:
            true_ids.append(parts[0])
            genres = parts[2].split(",")
            true_labels.append(genres)

# --- Load predicted genres ---
pred_ids = []
pred_labels = []

with open("predicted_genres.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ::: ")
        if len(parts) == 2:
            pred_ids.append(parts[0])
            pred_labels.append(parts[1].split(",") if parts[1] else [])

# --- Ensure matching order ---
id_to_pred = dict(zip(pred_ids, pred_labels))
pred_labels_aligned = [id_to_pred.get(id_, []) for id_ in true_ids]

# --- MultiLabelBinarizer on both sets ---
mlb = MultiLabelBinarizer()
Y_true = mlb.fit_transform(true_labels)
Y_pred = mlb.transform(pred_labels_aligned)

# --- Evaluation metrics ---
print("📊 Evaluation Metrics:")
print("Hamming Loss:", hamming_loss(Y_true, Y_pred))
print("Subset Accuracy:", accuracy_score(Y_true, Y_pred))
print("Micro F1 Score:", f1_score(Y_true, Y_pred, average='micro'))
print("Macro F1 Score:", f1_score(Y_true, Y_pred, average='macro'))

# --- Genre Distribution Visualization ---
flat_labels = [genre for sublist in true_labels for genre in sublist]
label_counts = Counter(flat_labels)

plt.figure(figsize=(12, 6))
plt.bar(label_counts.keys(), label_counts.values())
plt.xticks(rotation=90)
plt.title("True Genre Distribution")
plt.xlabel("Genre")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
