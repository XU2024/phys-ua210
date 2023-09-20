# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import timeit
## P2
L=100
e = 1.602*10**(-19)
ep = 8.854*10**(-12)
beta = e/(4*np.pi*ep)
V_total = 0
a = 2.82*10**(-10) ##0.282nm
for i in range(-L,L+1):
    for j in range(-L, L+1):
        for k in range(-L, L+1):
            if i==0 and j==0 and k==0:
                continue  ## skip i+j=k=0
            elif abs(i+j+k) % 2 == 0:
                V_total = V_total+ beta/(a*np.sqrt(i**2+j**2+k**2))
            else:
                V_total = V_total- beta/(a*np.sqrt(i**2+j**2+k**2))
M = (V_total/beta)*a
print(M)

testcode='''
import numpy as np
L=10
e = 1.602*10**(-19)
ep = 8.854*10**(-12)
beta = e/(4*np.pi*ep)
V_total = 0
a = 2.82*10**(-10) ##0.282nm
for i in range(-L,L+1):
    for j in range(-L, L+1):
        for k in range(-L, L+1):
            if i==0 and j==0 and k==0:
                continue  ## skip i+j=k=0
            elif abs(i+j+k) % 2 == 0:
                V_total = V_total+ beta/(a*np.sqrt(i**2+j**2+k**2))
            else:
                V_total = V_total- beta/(a*np.sqrt(i**2+j**2+k**2))
M = (V_total/beta)*a  '''
print(timeit.timeit(stmt=testcode))

L=10
e = 1.602*10**(-19)
ep = 8.854*10**(-12)
beta = e/(4*np.pi*ep)
V_total = 0
a = 2.82*10**(-10) ##0.282nm

-L <= int(i)<=L
-L<=int(j) <=L
-L<= int (k) <=L
while(i!=0 and j!=0 and k!=0):
  if abs(i+j+k) % 2 == 0:
    V_total = V_total+ beta/(a*np.sqrt(i**2+j**2+k**2))
  else:
    V_total = V_total- beta/(a*np.sqrt(i**2+j**2+k**2))
M = (V_total/beta)*a
print(M)

##3
import numpy as np
import matplotlib.pyplot as plt
import cmath

graph = np.zeros((100, 100)) ## the matrix of 100*100
def Mandelbrost(C):
  z=0
  N=0
  while abs(z)<=2 and N<=100:
    z = z**2+C
    N += 1
  return N    ## the number of iteration when it stops


x = np.linspace(-2, 2, 100) ## width = 100
y = np.linspace(-2, 2, 100) ## height = 100
for i in range(100):
  for j in range(100):
    c = complex(x[i], y[j])
    graph[i, j] = Mandelbrost(c)

plt.imshow(graph.T, extent=(-2, 2, -2, 2)) ## the size
plt.colorbar()
plt.title('Mandelbrot test')
plt.xlabel('real')
plt.ylabel('imaginary')
plt.show()

##P3
from matplotlib.font_manager import X11FontDirectories
import numpy as np
import matplotlib.pyplot as plt
## a)
a = np.float(input('Value of a:'))
b = np.float(input('Value of b:'))
c = np.float(input('Value of a:'))

x1= (-b+np.sqrt(b**2-4*a*c))/(2*a)
x2= (-b-np.sqrt(b**2-4*a*c))/(2*a)
print(x1, x2)

## b)
X1 = (2*c)/(-b-np.sqrt(b**2-4*a*c))
X2 = (2*c)/(-b+np.sqrt(b**2-4*a*c))
print(X1, X2)
## c)
xn1 = x1
xn2 = X2
print(xn1, xn2)
