import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# --------------------------------
# 1. Load dataset
# --------------------------------

data = load_breast_cancer()


# --------------------------------
# 2. Create DataFrame
# --------------------------------

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target


# --------------------------------
# 3. Check data
# --------------------------------

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())


# --------------------------------
# 4. Remove missing values
# --------------------------------

df = df.dropna()


# --------------------------------
# 5. Separate X and y
# --------------------------------

X = df.drop("target", axis=1)
y = df["target"]


# --------------------------------
# 6. Train/test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# 7. Standardization
# --------------------------------

X_train_mean = X_train.mean()
X_train_std = X_train.std()

X_train = (
    X_train - X_train_mean
) / X_train_std

X_test = (
    X_test - X_train_mean
) / X_train_std


# --------------------------------
# 8. Convert to NumPy
# --------------------------------

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# --------------------------------
# 9. Initialize parameters
# --------------------------------

b0 = 0

weights = np.zeros(
    X_train.shape[1]
)


# --------------------------------
# 10. Sigmoid function
# --------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --------------------------------
# 11. Hyperparameters
# --------------------------------

learning_rate = 0.01
iterations = 1000


# --------------------------------
# 12. Gradient Descent
# --------------------------------

for i in range(iterations):

    # Linear equation
    z = b0 + np.dot(
        X_train,
        weights
    )

    # Probability
    probabilities = sigmoid(z)

    # Error
    errors = probabilities - y_train

    # Gradients
    gradient_weights = (
        np.dot(
            X_train.T,
            errors
        ) / len(X_train)
    )

    gradient_b0 = np.mean(errors)

    # Update
    weights = (
        weights
        - learning_rate * gradient_weights
    )

    b0 = (
        b0
        - learning_rate * gradient_b0
    )


# --------------------------------
# 13. Test predictions
# --------------------------------

z_test = b0 + np.dot(
    X_test,
    weights
)

probabilities = sigmoid(z_test)

y_pred = (
    probabilities >= 0.5
).astype(int)


# --------------------------------
# 14. Accuracy
# --------------------------------

accuracy = np.mean(
    y_pred == y_test
)

print("\nAccuracy:", accuracy)


# --------------------------------
# 15. Confusion Matrix
# --------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------
# 16. Parameters
# --------------------------------

print("\nIntercept:", b0)

print("\nWeights:")
print(weights)

#using library
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# --------------------------------
# 1. Load dataset
# --------------------------------

data = load_breast_cancer()


# --------------------------------
# 2. Create DataFrame
# --------------------------------

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target


# --------------------------------
# 3. Check data
# --------------------------------

print(df.head())
print(df.shape)
print(df.isnull().sum())


# --------------------------------
# 4. Remove missing values
# --------------------------------

df = df.dropna()


# --------------------------------
# 5. Separate X and y
# --------------------------------

X = df.drop("target", axis=1)
y = df["target"]


# --------------------------------
# 6. Train/test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# 7. Feature scaling
# --------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# --------------------------------
# 8. Create model
# --------------------------------

model = LogisticRegression()


# --------------------------------
# 9. Train
# --------------------------------

model.fit(X_train, y_train)


# --------------------------------
# 10. Predictions
# --------------------------------

y_pred = model.predict(X_test)


# --------------------------------
# 11. Probabilities
# --------------------------------

y_probability = model.predict_proba(X_test)


# --------------------------------
# 12. Accuracy
# --------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)


# --------------------------------
# 13. Confusion Matrix
# --------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------
# 14. Classification Report
# --------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)
