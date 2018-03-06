import math

def sigmoid(z):
    return 1/(1+math.e**(-z))

x1=[]
x2=[]
x3=[]
y=[]

with open('/Users/xiongconghao/Downloads/breast-cancer-train.csv', 'r') as f:
    f.readline()
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
alpha=0.00001
for times in range(1,50):
    dw1=0.0
    dw2=0.0
    dw3=0.0
    db=0.0
    J=0.0
    for i in range(0,length-1):
        z=w1*x1[i]+w2*x2[i]+w3*x3[i]+b
        a=sigmoid(z)
        dz=a-y[i]
        J+=-(y[i]*math.log(math.e,a)+(1-y[i])*math.log(math.e,(1-a)))
        dw1+=x1[i]*dz
        dw2+=x2[i]*dz
        dw3+=x3[i]*dz
        db+=dz
    J/=length
    dw1/=length
    dw2/=length
    dw3/=length
    db/=length
    w1-=alpha*dw1
    w2-=alpha*dw2
    w3-=alpha*dw3
    b-=alpha*db

while int(input("Do you want to continue test?"))==1:
    test1=float(input("Input tumor thickness:"))
    test2=float(input("Input cell size:"))
    test3=float(input("Input final para:"))
    print(sigmoid(w1*test1+w2*test2+w3*test3+b))
