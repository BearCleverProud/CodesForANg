import math
import numpy as np
import time

start = time.clock()

def sigmoid(z):
    return 1/(1+math.e**(-z))

def costfunction(y,a):
    return (y-a)**2

x1=[]
x2=[]
x3=[]
y=[]

with open('breast-cancer-train.csv', 'r') as f:
    line=f.readline()
    while line:
        line=line.strip("\r\n")
        line=line.split(',')
        x1.append(int(line[0]))
        x2.append(int(line[1]))
        x3.append(int(line[2]))
        y.append(int(line[3]))
        line=f.readline()

length=len(x1)
x1=np.array(x1)
x2=np.array(x2)
x3=np.array(x3)
y=np.array(y).reshape(length,1)
w=np.array([0.0,0.0,0.0]).T.reshape(1,3)
b=0.0
J=10.0
alpha=0.65
X=np.array([x1,x2,x3]).T
times=0
accuracy=0.04

print("Calculating, please wait for some time")
while J>accuracy:
    dw=np.array([0.0,0.0,0.0]).T.reshape(1,3)
    db=0.0
    J=0
    z=np.dot(X,w.T)+b
    a=sigmoid(z)
    dz=(a-y)*2*a*(1-a)/length
    J=np.sum(costfunction(a,y))/length
    dw=np.sum(np.dot(X.T,dz),axis=1,keepdims=True).T/length
    db=np.sum(dz)/length
    w-=alpha*dw
    b-=alpha*db
    times+=1

print("Calculated times:",times)
print("Parameters are shown below:")
print(w,b)
elapsed = (time.clock() - start)
print("Time used:",elapsed)

while int(input("Do you want to test? Input 1:"))==1:
    test1=float(input("Input first para:"))
    test2=float(input("Input second para:"))
    test3=float(input("Input final para:"))
    test=np.array([test1,test2,test3]).T.reshape(3,1)
    result=((sigmoid(np.dot(w,test)+b)))
    print(result)
