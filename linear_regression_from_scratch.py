import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# 1. Load data and select one feature (BMI)
data = load_diabetes()
X = data.data[:, 2].reshape(-1, 1)
y = data.target

# 2. Scale the feature and flatten to 1D
X = StandardScaler().fit_transform(X).ravel()
n = len(y)

# 3. Initialise parameters and hyperparameters
w, b = 0.0, 0.0
lr0, decay, epochs = 0.1, 0.001, 1000
cost_history = []

# 4. Gradient descent with learning rate decay
for epoch in range(epochs):
    y_pred = w * X + b
    error = y_pred - y

    dw = (2 / n) * np.sum(error * X)
    db = (2 / n) * np.sum(error)

    lr = lr0 / (1 + decay * epoch)
    w = w - lr * dw
    b = b - lr * db

    cost_history.append(np.mean(error ** 2))

# 5. Final parameters and MSE
final_mse = np.mean((w * X + b - y) ** 2)
print(f"Learned w = {w:.4f}")
print(f"Learned b = {b:.4f}")
print(f"Final MSE = {final_mse:.4f}")

# 6. Plot cost vs epochs
plt.plot(cost_history)
plt.xlabel("Epoch")
plt.ylabel("Cost (MSE)")
plt.title("Cost vs Epochs")
plt.grid(True)
plt.show()

