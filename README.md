# machine-learning
1.linear regression 
-It is used for continous numerical value.
- Equation of straight line : y = mx + b
- y^ = wx + b
- The error = y - y^
- mean squared function : mse = 1/n summation (y - y^)
- Linear regression is used to find a line that minimizes MSE
- X = information we know , y = information we are trying to predict
- to make to array = np.array([])
- to find mean = np.mean(squared_errors)
- Gradient descent algorithm : Repeatedly adjust the model parameters in the direction that decreases the loss
- learning rate : controls how big our steps are
- gradient descent tells the direction in which loss increases the faster
- so we move in opposite direction the gradient
- <img width="161" height="113" alt="image" src="https://github.com/user-attachments/assets/e233b801-4465-406f-81b1-0dc8f97bbf2a" />
2.Polynomial regression
- Linear regression tries to fit in a straight line
- Polynomial Regression is still linear in its coefficients; it simply uses polynomial features of the input.
3.Logistic regression
- we use to predict class or category
- We use this when we want a probability between 0 and 1
- <img width="321" height="71" alt="image" src="https://github.com/user-attachments/assets/c34e9b60-a2b9-43f2-8c69-f0bc8994499a" />
- stratify = y  is used to keep the class balance when splitting
- in logistic regression , we use feature scaling

3.Decision Tree
import matplotlib.pyplot as plt

from sklearn.tree import ________, ________


model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

plt.figure(figsize=(15, 10))

________(
    model,
    feature_names=iris.________,
    class_names=iris.________,
    filled=True,
    rounded=True
)

plt.________("Decision Tree")
plt.________()
Answer:
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree


model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)

plt.title("Decision Tree")
plt.show()

