import math
import numpy as np
import time

start = time.clock()

def sigmoid(z):
    return 1/(1+math.e**(-z))

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
initialScale=0.01

x1=np.array(x1).reshape(length,1)
x2=np.array(x2).reshape(length,1)
x3=np.array(x3).reshape(length,1)
y=np.array(y).reshape(length,1)
W1=np.random.randn(4,3)*initialScale
b1=np.random.randn(4,1)
X=np.array((x1,x2,x3)).reshape(3,length)

W2=np.random.randn(1,4)*initialScale
b2=0

alpha=0.2
J=10
accuracy=0.04

while J>accuracy:
    Z1=np.dot(W1,X)+b1
    A1=sigmoid(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)

    dZ2=(A2-y.T)*2*A2*(1-A2)/length
    dw2=np.dot(dZ2,A1.T)
    db2=np.sum(dZ2)
    dZ1=np.dot(W2.T,dZ2)*(A1*(1-A1))*Z1/length
    dw1=np.dot(dZ1,X.T)
    db1=np.sum(dZ1,axis=1,keepdims=True)

    J=np.sum((A2-y.T)**2)/length

    W2-=dw2*alpha
    b2-=db2*alpha
    W1-=dw1*alpha
    b1-=db1*alpha
    print(J)

elapsed = (time.clock() - start)
print("Time used:",elapsed)

while int(input("Do you want to test? Input 1:"))==1:
    test1=float(input("Input first para:"))
    test2=float(input("Input second para:"))
    test3=float(input("Input final para:"))
    test=np.array([test1,test2,test3]).T.reshape(3,1)
    hiddenOutput=np.dot(W1,test)+b1
    hiddenOutput=sigmoid(hiddenOutput)
    finalOutput=np.dot(W2,hiddenOutput)+b2
    finalOutput=sigmoid(finalOutput)
    print(finalOutput)
