import numpy as np
import matplotlib.pyplot as plt
import random
import pandas

np.random.seed(12)
num = 500

d1 = np.random.multivariate_normal([0,0],[[1,.75],[.75,1]],num)
d2 = np.random.multivariate_normal([1,4],[[1,.75],[.75,1]],num)

#print(d1)
#print(x2)
X = np.vstack((d1,d2)).astype(np.float32)
#Y = np.hstack((np.zeros(num),np.ones(num)))
Y = np.hstack((np.ones(num),-np.ones(num)))

#plt.figure(1)
#plt.scatter(d1[:,0],d1[:,1],s=30,c='blue',marker='o',alpha=0.5,label='C1')
#plt.scatter(d2[:,0],d2[:,1],s=30,c='red',marker='x',alpha=0.5,label='C2')
#plt.show()

def sign(v):
    if v > 0:
        return 1
    else:
        return -1

def train(train_num, train_datas, lr): 
    w = [0, 0]
    b = 0
    for i in range(train_num): 
        x = random.choice(train_datas) 
       
        x1, x2, y = x
        if (y * sign((w[0] * x1 + w[1] * x2 + b)) <=0):
            w[0] += lr * y * x1
            w[1] += lr * y * x2
            b += lr * y
    return w, b

def plot_points(train_datas, w, b):
    plt.figure(2)
    
    x1 = np.linspace(-4, 4, 200)
    x2 = -(b + w[0] * x1)/w[1]
    plt.plot(x1, x2, color='r', label='y1 data')
    datas_len = len(train_datas)
    for i in range(datas_len):
        if train_datas[i][-1] == 1:
            plt.scatter(train_datas[i][0], train_datas[i][1], s=50)
        else:
            plt.scatter(train_datas[i][0], train_datas[i][1], marker='x', s=50)
    plt.show()

train_datas = np.column_stack((X,Y))

w, b = train(train_num=5000, train_datas=train_datas, lr=0.01)
plot_points(train_datas, w, b)
