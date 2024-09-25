Implementation-of-Logistic-Regression-Using-Gradient-Descent
AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
Initialize Parameters: Start by initializing the parameters (weights) theta with random values or zeros.
Compute Sigmoid Function: Define the sigmoid function that maps any real-valued number to a value between 0 and 1.
Compute Loss Function: Define the loss function, which measures the error between the predicted output and the actual output.
Gradient Descent Optimization: Implement the gradient descent algorithm to minimize the loss function. In each iteration, compute the gradient of the loss function with respect to the parameters (theta), and update the parameters in the opposite direction of the gradient to minimize the loss. 5.Iterate Until Convergence: Repeat the gradient descent steps for a predefined number of iterations or until convergence criteria are met. Convergence can be determined when the change in the loss function between iterations becomes very small or when the parameters (theta) stop changing significantly.
Program:
/* Program to implement the the Logistic Regression Using Gradient Descent. Developed by: k vishwa RegisterNumber: 212223080061 */


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y = Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
Output:
Dataset

image

Data types

image

New dataset

image

Y values

image

Accuracy

image

Y pred

image

New Y

image

image

image

Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
