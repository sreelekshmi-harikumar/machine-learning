import pandas as pd
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Load dataset
data = load_diabetes()


# Create DataFrame
df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target


# Check dataset
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())


# Remove missing values
df = df.dropna()


# X and y
X = df[["bmi"]]
y = df["target"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create polynomial features
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)


# Linear Regression
model = LinearRegression()

model.fit(
    X_train_poly,
    y_train
)


# Predictions
y_train_pred = model.predict(X_train_poly)

y_test_pred = model.predict(X_test_poly)


# Evaluation

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


# Results
print("\nTraining MAE:", train_mae)
print("Testing MAE:", test_mae)

print("\nTraining MSE:", train_mse)
print("Testing MSE:", test_mse)

print("\nTraining RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)

print("\nTraining R²:", train_r2)
print("Testing R²:", test_r2)

import pandas as pd
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
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
# 2. DataFrame
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
# 5. X and y
# --------------------------------

X = df[["bmi"]]
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

X_train = X_train.to_numpy().flatten()
X_test = X_test.to_numpy().flatten()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# --------------------------------
# 8. Create polynomial features
# --------------------------------

X_train_poly = np.column_stack([
    X_train,
    X_train ** 2
])

X_test_poly = np.column_stack([
    X_test,
    X_test ** 2
])


# --------------------------------
# 9. Initialize parameters
# --------------------------------

b0 = 0

weights = np.zeros(
    X_train_poly.shape[1]
)


# --------------------------------
# 10. Hyperparameters
# --------------------------------

learning_rate = 0.01
iterations = 10000


# --------------------------------
# 11. Gradient Descent
# --------------------------------

for i in range(iterations):

    # Prediction
    y_pred = b0 + np.dot(
        X_train_poly,
        weights
    )

    # Error
    errors = y_train - y_pred

    # Gradient
    gradient_b0 = (
        (-2 / len(X_train_poly))
        * np.sum(errors)
    )

    gradient_weights = (
        (-2 / len(X_train_poly))
        * np.dot(
            X_train_poly.T,
            errors
        )
    )

    # Update
    b0 = (
        b0
        - learning_rate * gradient_b0
    )

    weights = (
        weights
        - learning_rate * gradient_weights
    )


# --------------------------------
# 12. Predictions
# --------------------------------

y_train_pred = b0 + np.dot(
    X_train_poly,
    weights
)

y_test_pred = b0 + np.dot(
    X_test_poly,
    weights
)


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

print("\nTraining R²:", train_r2)
print("Testing R²:", test_r2)
