import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Load predictions
pred_df = pd.read_csv("predictions.txt", sep="\t", names=["ID", "PREDICTED_GENRE"])
pred_df["ID"] = pred_df["ID"].astype(str)

# Load ground truth
true_df = pd.read_csv("test_data_solution.txt", sep=" ::: ", engine="python", names=["ID", "TITLE", "GENRE", "DESCRIPTION"])
true_df["ID"] = true_df["ID"].astype(str)

# Merge on ID
merged_df = pd.merge(true_df, pred_df, on="ID")

# Convert to lists for multi-label support (if needed)
y_true = merged_df["GENRE"].apply(lambda x: x.split(",") if isinstance(x, str) else ["None"])
y_pred = merged_df["PREDICTED_GENRE"].apply(lambda x: x.split(",") if isinstance(x, str) else ["None"])

from sklearn.preprocessing import MultiLabelBinarizer

# Fit on all genres from the union of true and predicted
mlb = MultiLabelBinarizer()
y_true_bin = mlb.fit_transform(y_true)
y_pred_bin = mlb.transform(y_pred)

# Evaluation
exact_match = accuracy_score(y_true_bin, y_pred_bin)
micro_f1 = f1_score(y_true_bin, y_pred_bin, average='micro')
macro_f1 = f1_score(y_true_bin, y_pred_bin, average='macro')

print(f"🎯 Exact Match Accuracy: {exact_match:.4f}")
print(f"🎯 Micro F1 Score: {micro_f1:.4f}")
print(f"🎯 Macro F1 Score: {macro_f1:.4f}")
