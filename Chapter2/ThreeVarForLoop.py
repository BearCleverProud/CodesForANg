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

w1=0.0
w2=0.0
w3=0.0
b=0.0
alpha=0.001
length=len(x1)
for i in range(0,length-1):
    z=w1*x1[i]+w2*x2[i]+w3*x3[i]+b
    a=sigmoid(z)
    l=a-y[i]
    print(l)
    w1-=alpha*l*x1[i]
    w2-=alpha*l*x2[i]
    w3-=alpha*l*x3[i]
    b-=alpha*l

print("w1="+str(w1))
print("w2="+str(w2))
print("w3="+str(w3))
print("b="+str(b))

while int(input("Do you want to continue test?"))==1:
    test1=float(input("Input tumor thickness:"))
    test2=float(input("Input cell size:"))
    test3=float(input("Input final para:"))
    print(sigmoid(w1*test1+w2*test2+w3*test3+b))
