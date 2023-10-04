# -*- coding: utf-8 -*-


##1
import math as math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad

def f(x):
  return x**4-2*x+1

def tp_integral(N):
  a=0
  b=2
  deltax = (b-a)/N
  Integral = 0

  for i in range(N+1): ## x0, x1...XN
    xi= a+deltax*i
    if i==0 or i==N:
      Integral += f(xi)
    else:
      Integral += 2*f(xi)

  Integral *= deltax/2
  return Integral
error = (tp_integral(20)-tp_integral(10))/3
realvalue =4.4
print(tp_integral(20))
print(tp_integral(10))
print(error)

##2 part(b)


def V(x):
    return x**4



def period(a, m, N):
    def density_function(x):
      p = np.sqrt(8*m)/np.sqrt(V(a)-V(x))
      return p

    ## using gaussian quadrature
    x_sample, weight = np.polynomial.legendre.leggauss(N)
    result = sum(weight * density_function(x_sample))
    return result

# Range of amplitudes
A = np.linspace(0, 2, 100)

periods = [period(a, 1, 20) for a in A]


plt.plot(A, periods)
plt.xlabel('Amplitude (a)')
plt.ylabel('Period (T)')
plt.title('Period vs. Amplitude')
plt.grid(True)
plt.show()

## (3)

#  a)
def Hermite_polynomial(n, x):
  if n==0:
    H = 1
  elif n==1:
    H = 2*x
  else:
    h0 =1 # intial value
    h1 =2*x
    for i in range(2, n+1):
      H = 2*x*h1-2*(i-1)*h0
      h0 = h1
      h1 = H
  return H
print(Hermite_polynomial(3, 4))

def harmonic_oscillator(n, x):
  f = []
  for i in range(n+1):
    fi = (np.e**(-x**2/2)*Hermite_polynomial(i, x))/(np.sqrt((2**i)*math.factorial(i)*np.sqrt(np.pi)))
    f.append(fi)
  return f
print(harmonic_oscillator(3,4))   ## values psi0(4), psi1(4), psi2(4), psi3(4)

x = np.linspace(-4, 4, 100)
y0 = (np.e**(-x**2/2)*Hermite_polynomial(0, x))/(np.sqrt((2**0)*math.factorial(0)*np.sqrt(np.pi)))
y1 = (np.e**(-x**2/2)*Hermite_polynomial(1, x))/(np.sqrt((2**1)*math.factorial(1)*np.sqrt(np.pi)))
y2 = (np.e**(-x**2/2)*Hermite_polynomial(2, x))/(np.sqrt((2**2)*math.factorial(2)*np.sqrt(np.pi)))
y3 = (np.e**(-x**2/2)*Hermite_polynomial(3, x))/(np.sqrt((2**3)*math.factorial(3)*np.sqrt(np.pi)))

plt.plot(x, y0, label = 'n=0')
plt.plot(x, y1, label = 'n=1')
plt.plot(x, y2, label = 'n=2')
plt.plot(x, y3, label = 'n=3')
plt.xlabel('x')
plt.ylabel('hn(x)')
plt.title('Wavefunction of quantum harmonic oscillator')
plt.grid(True)
plt.legend
plt.show()

#  b)
x1 = np.linspace(-10, 10, 100)
y30 = (np.e**(-x**2/2)*Hermite_polynomial(30, x))/(np.sqrt((2**30)*math.factorial(30)*np.sqrt(np.pi)))
plt.plot(x1, y30, label = 'n=30')
plt.xlabel('x')
plt.ylabel('hn(x)')
plt.title('Wavefunction of quantum harmonic oscillator')
plt.grid(True)
plt.legend
plt.show()

# c)
n= 5
# wavefunction with n=5
def y5(x):
  y = (np.e**(-x**2/2)*Hermite_polynomial(5, x))/(np.sqrt((2**5)*math.factorial(5)*np.sqrt(np.pi)))
  return y    ## define the wavefunction with n=5

xs1, w1 = np.polynomial.legendre.leggauss(100) ## Gaussian quadrature with 100 points, w=weights; new x is the specifc x points
p1 =  (xs1**2)*y5(xs1)**2 ## mean desnity
result1 = np.sqrt(sum(w1 * p1))

print(result1)

# d)
n= 5
# wavefunction with n=5

xs2, w2 = np.polynomial.hermite.hermgauss(100) ## Gauss-Hermite quadrature with 100 points, w=weights; new x is the specifc x points
p2 =  (xs2**2)*y5(xs2)**2 ## mean density
result2 = np.sqrt(sum(w2 * p2))

print(result2)
