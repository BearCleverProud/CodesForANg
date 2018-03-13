import math

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
w1=0.0
w2=0.0
w3=0.0
b=0.0
alpha=0.002978971912446
for times in range(1,100):
    dw1=0.0
    dw2=0.0
    dw3=0.0
    db=0.0
    for i in range(0,length):
        z=w1*x1[i]+w2*x2[i]+w3*x3[i]+b
        a=sigmoid(z)
        dz=a-y[i]
        dw1+=dz*x1[i]
        dw2+=dz*x2[i]
        dw3+=dz*x3[i]
        db+=dz
    dw1=dw1/float(length)
    dw2=dw2/float(length)
    dw3=dw3/float(length)
    db=db/float(length)
    w1=w1-alpha*dw1
    w2=w2-alpha*dw2
    w3=w3-alpha*dw3
    b=b-alpha*db

resultOffSet=-0.683878244925
power=2
print(w1,w2,w3,b)
while int(input("Do you want to continue test?:"))==1:
    test1=float(input("Input first para:"))
    test2=float(input("Input second para:"))
    test3=float(input("Input final para:"))
    print(((sigmoid(w1*test1+w2*test2+w3*test3+b)+resultOffSet)/(1+resultOffSet))**(power))
