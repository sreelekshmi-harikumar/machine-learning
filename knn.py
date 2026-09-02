import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# --------------------------------------------------
# 1. Load dataset
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
# 4. Check missing values
# --------------------------------------------------

print("Missing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 5. Remove missing values
# --------------------------------------------------

df = df.dropna()


# --------------------------------------------------
# 6. Separate features and target
# --------------------------------------------------

X = df.drop('target', axis=1)
y = df['target']


# --------------------------------------------------
# 7. Split data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 8. Feature scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------------
# 9. Create KNN model
# --------------------------------------------------

model = KNeighborsClassifier(
    n_neighbors=5
)


# --------------------------------------------------
# 10. Train model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 11. Predict
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 12. Accuracy
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)


# --------------------------------------------------
# 13. Confusion Matrix
# --------------------------------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 14. Classification Report
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)
