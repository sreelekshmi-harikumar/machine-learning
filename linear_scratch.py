import pandas as pd
from sklearn.datasets import load_diabetes
data = load_diabetes()
df = pd.DataFrame(
  data.data,
  data.feature_names
)
df["target"] = data.target
print(df.head()) #to print a few rows
print(df.shape) #(442,11) - 442 examples and 11 columns
print(df.info())
print(df.isnull().sum()) #total sum for missing values
df= df.dropna() #to remove missing values
X = df.drop("target",axis = 1)
y = df.["target"]
X_one = X.iloc[:,0]
'''
X = data.data #information we use to make predictions 
y = data.target #the values we need to predict
print(X.shape)
print(y.shape) #to see the dimension
#(442,10) - 442 examples and 10 features for each
X_one = X[:,0] #for every example give me row 0
'''
