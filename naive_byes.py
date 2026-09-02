import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# 1. Load the Iris dataset
iris = load_iris()


# 2. Create a DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)


# 3. Add the target column
df['target'] = iris.target


# 4. Display first 5 rows
print("First 5 rows:")
print(df.head())


# 5. Check for missing values
print("\nMissing values:")
print(df.isnull().sum())


# 6. Remove rows containing missing values
df = df.dropna()


# 7. Check missing values again
print("\nMissing values after removal:")
print(df.isnull().sum())


# 8. Display DataFrame information
print("\nDataset shape:")
print(df.shape)


# 9. Separate features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']


# 10. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 11. Create Naive Bayes model
model = GaussianNB()


# 12. Train the model
model.fit(X_train, y_train)


# 13. Make predictions
y_pred = model.predict(X_test)


# 14. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)


# 15. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 16. Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 17. Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\nPrediction for new flower:")
print(prediction)

print("Predicted flower name:")
print(iris.target_names[prediction])
