import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# --------------------------------------------------
# 1. Load the Iris dataset
# --------------------------------------------------

iris = load_iris()


# --------------------------------------------------
# 2. Create DataFrame
# --------------------------------------------------

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)


# --------------------------------------------------
# 3. Add target column
# --------------------------------------------------

df['target'] = iris.target


# --------------------------------------------------
# 4. Display first 5 rows
# --------------------------------------------------

print("First 5 rows:")
print(df.head())


# --------------------------------------------------
# 5. Check missing values
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 6. Remove missing values
# --------------------------------------------------

df = df.dropna()


# --------------------------------------------------
# 7. Check missing values again
# --------------------------------------------------

print("\nMissing values after removal:")
print(df.isnull().sum())


# --------------------------------------------------
# 8. Display dataset information
# --------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())


# --------------------------------------------------
# 9. Separate features and target
# --------------------------------------------------

X = df.drop('target', axis=1)
y = df['target']


# --------------------------------------------------
# 10. Split data into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# --------------------------------------------------
# 11. Create Decision Tree model
# --------------------------------------------------

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=3
)


# --------------------------------------------------
# 12. Train the model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 13. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 14. Calculate accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)


# --------------------------------------------------
# 15. Confusion Matrix
# --------------------------------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 16. Classification Report
# --------------------------------------------------

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))


# --------------------------------------------------
# 17. Predict a new flower
# --------------------------------------------------

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\nPrediction:")
print(prediction)

print("Predicted flower:")
print(iris.target_names[prediction][0])


# --------------------------------------------------
# 18. Plot the Decision Tree
# --------------------------------------------------

plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)

plt.title("Decision Tree - Iris Dataset")

plt.show()
