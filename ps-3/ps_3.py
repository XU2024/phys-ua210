
## P1
import numpy as np
from numpy import zeros

def f(x):
  y = x*(x-1)
  return y

def derivative(x, delta):
  z = (f(x+delta)-f(x))/delta
  return z
print(f(1), derivative(1, 0.01))
print(f(1), derivative(1, 10**(-4)))
print(f(1), derivative(1, 10**(-6)))
print(f(1), derivative(1, 10**(-8)))
print(f(1), derivative(1, 10**(-10)))
print(f(1), derivative(1, 10**(-12)))
print(f(1), derivative(1, 10**(-14)))

real = 1    ## 2x-1

## P2
import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt
import time
Size = []
explicit_operation=[]
dot_operation = []

def operation(N):
  C = zeros([N, N], float)
  operation = 0
  for i in range(N):
    for j in range(N):
      for k in range(N):
        C[i,j] += A[i,k]*B[k,j]
        operation +=1  ## number of addtions
        operation +=1  ## number of multiplication
  return operation

def dot(N):
  return (2 * N**3) - N**2 ## got from the internet

for N in range(10, 101, 10): ## matrix from 10*10 to 100*100
  Size.append(N) ## put each N into the Size array
  explicit_operation.append(operation(N))
  dot_operation.append(dot(N))


plt.plot(Size, explicit_operation,'b' )
plt.xlabel('Size (N)')
plt.ylabel('Operations(flops)')
plt.show()


plt.xlabel('Size (N)')
plt.plot(Size, dot_operation,'r' )
plt.ylabel('Operations(flops)')
plt.show()

## P2 Another method using time
Size1 =[]
explicit_t = []
dot_t = []

for N in range(10, 101, 10):
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)

    start_time = time.time() ## meausre the time for explicit one
    C = np.zeros((N, N), float)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    T_e = time.time() - start_time

    start_time = time.time()## meausre the time for dot one
    C = np.dot(A, B)
    T_d = time.time() - start_time


    Size1.append(N)
    explicit_t.append(T_e)
    dot_t.append(T_d)


plt.plot(Size1, explicit_t, marker = 'o', label='Explicit')
plt.plot(Size1, dot_t, marker = 'o', label='dot')
plt.xlabel('Size (N)')
plt.ylabel('Time (seconds)')
plt.title('Multiplication Computation Time vs.  Size')
plt.legend()
plt.grid(True)
plt.show()

##P3
from random import random

h = 1 ## time step
Bi213=10000
Pb209=0
Tl209=0
Bi209=0

p_Pb209 = 1-2**(-h/(3.3*60)) ## use the equation to get the probability
p_Tl209 = 1-2**(-h/(2.2*60))
p_Bi213 = 1-2**(-h/(46*60))


t = np.arange(0,20000,1)
Bi213c= []
Pb209c= []
Tl209c= []
Bi209c= []

for i in t:
  Bi213c.append(Bi213)
  Pb209c.append(Pb209)
  Tl209c.append(Tl209)
  Bi209c.append(Bi209)

  for j in range(Pb209):
    if random()<p_Pb209:
      Pb209-=1
      Bi209+=1

  for j in range(Tl209):
    if random()<p_Tl209 :
      Tl209 -=1
      Pb209 +=1

  for j in range(Bi213):
    if random()<p_Bi213 :
      Bi213 -=1
      if random()<0.00209:
        Tl209 +=1
      else:
        Pb209 +=1

plt.plot(t,Pb209c,label='Pb209')
plt.plot(t,Tl209c,label='Tl209')
plt.plot(t,Bi213c,label='Bi213')
plt.plot(t,Bi209c,label='Bi209')
plt.legend()
plt.xlabel('time(s)')
plt.ylabel('# atoms')
plt.show()

##p4
from numpy.random import random

N = 1000

lamda= np.log(2)/(3.053*60) ## decat constnat of Tl

x = random(N) ## get a number of decay randomly

t = -1/lamda*log(1-x) ## equation to get t for the specific number of decay
t = sort(t)

decay = arange(1,1001)
Nodecay = 1000-decay  ## arrat for nondecay

plt.plot(t, Nodecay)
plt.xlabel('time(s)')
plt.ylabel('# not decayed atoms')
plt.show()
