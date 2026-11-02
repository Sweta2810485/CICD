
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    f1_score,
    recall_score,
    accuracy_score
)

sns.set(style="white")

# Load Dataset
dataset = pd.read_csv("iris.csv")

# Clean Column Names
dataset.columns = [
    col.strip().lower().replace(" (cm)", "").replace(" ", "_")
    for col in dataset.columns
]

# Check target column
if "target" not in dataset.columns:
    raise ValueError("Target column not found in iris.csv")

# Feature Engineering
dataset["sepal_length_width_ratio"] = (
    dataset["sepal_length"] / dataset["sepal_width"]
)

dataset["petal_length_width_ratio"] = (
    dataset["petal_length"] / dataset["petal_width"]
)

# Select Features
dataset = dataset[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "sepal_length_width_ratio",
        "petal_length_width_ratio",
        "target"
    ]
]

# Split Dataset
train_data, test_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=44,
    stratify=dataset["target"]
)

# Prepare Training and Testing Data
X_train = train_data.drop("target", axis=1).values.astype("float32")
y_train = train_data["target"].values.astype("int32")

X_test = test_data.drop("target", axis=1).values.astype("float32")
y_test = test_data["target"].values.astype("int32")

# Class Names
target_name = ["setosa", "versicolor", "virginica"]

# --------------------------------
# Logistic Regression
# --------------------------------

logreg = LogisticRegression(
    C=0.0001,
    solver="lbfgs",
    max_iter=1000
)

logreg.fit(X_train, y_train)

predictions_lr = logreg.predict(X_test)

cm_lr = confusion_matrix(y_test, predictions_lr)

f1_lr = f1_score(y_test, predictions_lr, average="micro")
prec_lr = precision_score(y_test, predictions_lr, average="micro")
recall_lr = recall_score(y_test, predictions_lr, average="micro")

train_acc_lr = accuracy_score(
    y_train, logreg.predict(X_train)
) * 100

test_acc_lr = accuracy_score(
    y_test, predictions_lr
) * 100

# --------------------------------
# Random Forest Classifier
# --------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=44
)

rf_model.fit(X_train, y_train)

predictions_rf = rf_model.predict(X_test)

cm_rf = confusion_matrix(y_test, predictions_rf)

f1_rf = f1_score(y_test, predictions_rf, average="micro")
prec_rf = precision_score(y_test, predictions_rf, average="micro")
recall_rf = recall_score(y_test, predictions_rf, average="micro")

train_acc_rf = accuracy_score(
    y_train, rf_model.predict(X_train)
) * 100

test_acc_rf = accuracy_score(
    y_test, predictions_rf
) * 100

# --------------------------------
# Confusion Matrix Plotting
# --------------------------------

def plot_cm(cm, target_name, title, filename, normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 7))

    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(target_name))

    plt.xticks(tick_marks, target_name, rotation=45)
    plt.yticks(tick_marks, target_name)

    thresh = cm.max() / 1.5

    for i, j in itertools.product(
        range(cm.shape[0]), range(cm.shape[1])
    ):
        plt.text(
            j,
            i,
            "{:0.4f}".format(cm[i, j]) if normalize else str(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()

    plt.ylabel("True Label")
    plt.xlabel(
        "Predicted Label\naccuracy={:0.4f}; misclass={:0.4f}".format(
            accuracy, misclass
        )
    )

    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()


# Logistic Regression Confusion Matrix
plot_cm(
    cm_lr,
    target_name,
    "Confusion Matrix - Logistic Regression",
    "ConfusionMatrix.png"
)

# Random Forest Confusion Matrix
plot_cm(
    cm_rf,
    target_name,
    "Confusion Matrix - Random Forest",
    "ConfusionMatrix_RF.png"
)

# --------------------------------
# Feature Importance
# --------------------------------

importances = rf_model.feature_importances_

labels = dataset.columns[:-1]

feature_df = pd.DataFrame({
    "feature": labels,
    "importance": importances
})

features = feature_df.sort_values(
    by="importance",
    ascending=False
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x="importance",
    y="feature",
    data=features
)

plt.xlabel("Importance", fontsize=14)
plt.ylabel("Feature", fontsize=14)

plt.title("Random Forest Feature Importances")

plt.tight_layout()

plt.savefig("FeatureImportance.png", dpi=120, bbox_inches="tight")
plt.close()

# --------------------------------
# Save Model Scores
# --------------------------------

with open("scores.txt", "w") as score:

    score.write("RANDOM FOREST CLASSIFIER\n")
    score.write("--------------------------------\n")

    score.write("Random Forest Train Accuracy: %.2f%%\n" % train_acc_rf)
    score.write("Random Forest Test Accuracy: %.2f%%\n" % test_acc_rf)
    score.write("F1 Score: %.2f%%\n" % (f1_rf * 100))
    score.write("Recall Score: %.2f%%\n" % (recall_rf * 100))
    score.write("Precision Score: %.2f%%\n" % (prec_rf * 100))

    score.write("\nLOGISTIC REGRESSION\n")
    score.write("--------------------------------\n")

    score.write("Logistic Regression Train Accuracy: %.2f%%\n" % train_acc_lr)
    score.write("Logistic Regression Test Accuracy: %.2f%%\n" % test_acc_lr)
    score.write("F1 Score: %.2f%%\n" % (f1_lr * 100))
    score.write("Recall Score: %.2f%%\n" % (recall_lr * 100))
    score.write("Precision Score: %.2f%%\n" % (prec_lr * 100))

print("Model Training Completed Successfully!")

print("Random Forest Accuracy:", test_acc_rf)
print("Logistic Regression Accuracy:", test_acc_lr)

print("Generated Files:")
print("1. scores.txt")
print("2. ConfusionMatrix.png")
print("3. ConfusionMatrix_RF.png")
print("4. FeatureImportance.png")
