import math

# find the zero point of 3+14t-5t^2
initvar=10
alpha=0.00001
error=(3+14*initvar-5*(initvar**2))**2/2
count=0
while error>0.00000000001 or error<-0.00000000001:
    initvar-=(-10*initvar+14)*(-5*initvar**2+14*initvar+3)*alpha
    error=(3+14*initvar-5*(initvar**2))**2/2
    count+=1
    print(initvar)

print(initvar)
print(count)
print(error)
