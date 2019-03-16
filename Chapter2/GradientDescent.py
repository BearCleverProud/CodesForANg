
# find the zero point of 3+14t-5t^2

x=2
alpha=0.001
error=(3+14*x-5*(x**2))**2
count=0
accuracy=0.00000000001
while error>accuracy or error<-accuracy:
    x-=(-10*x+14)*(-5*x**2+14*x+3)*alpha*2
    error=(3+14*x-5*(x**2))**2
    count+=1

print(x)
print(count)
print(error)
