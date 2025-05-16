from sklearn.datasets import fetch_california_housing
import numpy as np
import torch
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
print(data.feature_names)
 
X, y = data.data, data.target
 
# train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

print(X_train)
 
