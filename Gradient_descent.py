import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

# Defince the functioin that help plotting and drawing the line

def plot_points(X,y):
    # in here We have two classes these are admitted and rejected 
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    admitted_x = [s[0][0] for s in admitted]
    admitted_y = [s[0][1] for s in admitted]
    rejected_x = [s[0][0] for s in rejected]
    rejected_y = [s[0][1] for s in rejected]
    plt.scatter(admitted_x,admitted_y,s=25,color='blue',edgecolor='k')
    plt.scatter(rejected_x,rejected_y,s=25,color='red',edgecolor='k')
def display (m,b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)

# Reading and plotting the data
data = pd.read_csv('C:/Users/thinkpad/Documents/data.csv', header=None)
X = np.array(data[[0,1]])
y = np.array(data[2])


#plot_points(X,y)
# plt.show()

# todo: implement the basic functions
# - sigmoid activation funciton
# - prediction 
# - error funtion 
# - update the weight for each one in the weights list
# - update the bias for each one in the bias list

def sigmoid(x):
    return (1/(1+np.exp(-x)))

# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features,weights)+bias)

# Error (log-loss) formula
def error_formula(y, output):
    return (-y*np.log(output) - (1 -y )*np.log(1-output))

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    out_put = output_formula(x,weights,bias)
    #err = error_formula(y , out_put)
    err = y - out_put
    weights = weights + learnrate*err*x 
    bias = bias + learnrate*err
    return weights, bias

# Training function 

