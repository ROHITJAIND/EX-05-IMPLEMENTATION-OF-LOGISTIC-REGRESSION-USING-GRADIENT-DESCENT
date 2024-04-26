# EX-05 Implementation of Logistic Regression Using Gradient Descent
### AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent. &emsp; &emsp; &emsp;&emsp;&emsp; &emsp;&emsp;&emsp;**DATE:**
### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
### Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary .
4. Calculate the y-prediction. 
<table>
<tr>
<th>
  
### Program:
</th>
<th>
  
### Output:
</th>
</tr>
<tr>
<td width=40%>
  
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
df = pd.read_csv('CSVs/Placement_Data5.csv')
df.head()
```

</td> 
<td>

![image](https://github.com/ROHITJAIND/EX-05-IMPLEMENTATION-OF-LOGISTIC-REGRESSION-USING-GRADIENT-DESCENT/assets/118707073/333b06e8-a2e0-4fe4-adf2-06b3f8f70b47)

</td>
</tr> 
</table>
<table>
<tr>
<td width=40%>
  
```Python
df = df.drop('sl_no',axis=1) 
df = df.drop('salary',axis=1) 
A=["gender","ssc_b","hsc_b","degree_t",
   "workex","specialisation","status","hsc_s"]
for i in A:
    df[i]=df[i].astype('category')
df.dtypes
```
</td> 
<td>

![image](https://github.com/ROHITJAIND/EX-05-IMPLEMENTATION-OF-LOGISTIC-REGRESSION-USING-GRADIENT-DESCENT/assets/118707073/9afe6ac5-a6a6-47e5-a730-9f22b9e09797)
</td>
</tr> 
</table>
<table>
<tr>
<td width=40%>
  
```Python
for j in A:
    df[j]=df[j].cat.codes
df
```

</td> 
<td>

![image](https://github.com/ROHITJAIND/EX-05-IMPLEMENTATION-OF-LOGISTIC-REGRESSION-USING-GRADIENT-DESCENT/assets/118707073/14fbe535-7625-45be-b3bb-07a855a6f67f)
</td>
</tr> 
</table>

<table>
<tr>
<td width=40%>
  
```Python
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
Y
```

</td> 
<td>

![image](https://github.com/ROHITJAIND/EX-05-IMPLEMENTATION-OF-LOGISTIC-REGRESSION-USING-GRADIENT-DESCENT/assets/118707073/a382bf2a-c63a-46a1-b008-a2b133148d2b)
</td>
</tr> 
</table>

<table>
<tr>
<td width=40%>
  
```Python
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
# Define the gradient descent algorithm.
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta
# Train the model.
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
# Make predictions.
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
# Evaluate the model.
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy) 
```

</td> 
<td>

![image](https://github.com/ROHITJAIND/EX-05-IMPLEMENTATION-OF-LOGISTIC-REGRESSION-USING-GRADIENT-DESCENT/assets/118707073/8ba05b46-055a-4b4c-9286-acee65e41498)

</td>
</tr> 
</table>

<table>
<tr>
<td width=40%>
  
```Python
print(y_pred)
```

</td> 
<td>

![image](https://github.com/ROHITJAIND/EX-05-IMPLEMENTATION-OF-LOGISTIC-REGRESSION-USING-GRADIENT-DESCENT/assets/118707073/973b0548-c081-4b7f-bdcf-3c339901e6e8)

</td>
</tr> 
</table>

<table>
<tr>
<td width=40%>
  
```Python
print(Y)
```

</td> 
<td>

![image](https://github.com/ROHITJAIND/EX-05-IMPLEMENTATION-OF-LOGISTIC-REGRESSION-USING-GRADIENT-DESCENT/assets/118707073/b91fd466-5a16-4f22-8248-c778d0fa9521)
</td>
</tr> 
</table>

<table>
<tr>
<td width=40%>
  
```Python
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])        
y_prednew = predict(theta, xnew)
print(y_prednew)
```

</td> 
<td>

![image](https://github.com/ROHITJAIND/EX-05-IMPLEMENTATION-OF-LOGISTIC-REGRESSION-USING-GRADIENT-DESCENT/assets/118707073/36066a97-efb0-4c11-acbe-56e08f3759ba)


</td>
</tr> 
</table>

<table>
<tr>
<td width=40%>
  
```Python
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])        
y_prednew = predict(theta, xnew)
print(y_prednew)
```
</td> 
<td>

![image](https://github.com/ROHITJAIND/EX-05-IMPLEMENTATION-OF-LOGISTIC-REGRESSION-USING-GRADIENT-DESCENT/assets/118707073/40f98a9f-469c-4abf-8d97-5c29804f3b13)
</td>
</tr> 
</table>

```
Developed By: ROHIT JAIN D 
Register No : 212222230120
```

### Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

