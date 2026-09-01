import pandas as pd
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------
# 1. Load dataset
# --------------------------------
data = load_diabetes()


# --------------------------------
# 2. Create DataFrame
# --------------------------------
df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target


# --------------------------------
# 3. Check the dataset
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
    random_state=42
)


# --------------------------------
# 7. Convert to NumPy
# --------------------------------
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# --------------------------------
# 8. Initialize parameters
# --------------------------------
b0 = 0
weights = np.zeros(X_train.shape[1])


# --------------------------------
# 9. Hyperparameters
# --------------------------------
learning_rate = 0.01
iterations = 1000


# --------------------------------
# 10. Train using Gradient Descent
# --------------------------------
for i in range(iterations):

    # Prediction
    y_pred = b0 + np.dot(X_train, weights)

    # Error
    errors = y_train - y_pred

    # Gradients
    gradient_b0 = (-2 / len(X_train)) * np.sum(errors)

    gradient_weights = (
        (-2 / len(X_train))
        * np.dot(X_train.T, errors)
    )

    # Update parameters
    b0 = b0 - learning_rate * gradient_b0

    weights = weights - learning_rate * gradient_weights


# --------------------------------
# 11. Training predictions
# --------------------------------
y_train_pred = b0 + np.dot(X_train, weights)


# --------------------------------
# 12. Testing predictions
# --------------------------------
y_test_pred = b0 + np.dot(X_test, weights)


# --------------------------------
# 13. Evaluation
# --------------------------------
train_mae = mean_absolute_error(
    y_train,
    y_train_pred
)

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

train_mse = mean_squared_error(
    y_train,
    y_train_pred
)

test_mse = mean_squared_error(
    y_test,
    y_test_pred
)

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

train_r2 = r2_score(
    y_train,
    y_train_pred
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)


# --------------------------------
# 14. Results
# --------------------------------
print("\nIntercept:", b0)
print("Weights:", weights)

print("\nTraining MAE:", train_mae)
print("Testing MAE:", test_mae)

print("\nTraining MSE:", train_mse)
print("Testing MSE:", test_mse)

print("\nTraining RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)

print("\nTraining R2:", train_r2)
print("Testing R2:", test_r2)

import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------
# 1. Load dataset
# --------------------------------
data = load_diabetes()


# --------------------------------
# 2. Create DataFrame
# --------------------------------
df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target


# --------------------------------
# 3. Check dataset
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
    random_state=42
)


# --------------------------------
# 7. Create model
# --------------------------------
model = LinearRegression()


# --------------------------------
# 8. Train model
# --------------------------------
model.fit(X_train, y_train)


# --------------------------------
# 9. Predictions
# --------------------------------
y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)


# --------------------------------
# 10. Evaluation
# --------------------------------
train_mae = mean_absolute_error(
    y_train,
    y_train_pred
)

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

train_mse = mean_squared_error(
    y_train,
    y_train_pred
)

test_mse = mean_squared_error(
    y_test,
    y_test_pred
)

train_rmse = train_mse ** 0.5
test_rmse = test_mse ** 0.5

train_r2 = r2_score(
    y_train,
    y_train_pred
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)


# --------------------------------
# 11. Results
# --------------------------------
print("\nIntercept:", model.intercept_)

print("\nCoefficients:", model.coef_)

print("\nTraining MAE:", train_mae)
print("Testing MAE:", test_mae)

print("\nTraining MSE:", train_mse)
print("Testing MSE:", test_mse)

print("\nTraining RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)

print("\nTraining R2:", train_r2)
print("Testing R2:", test_r2)

