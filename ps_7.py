# -*- coding: utf-8 -*-




import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import scipy.optimize as optimize
from jax import grad, hessian
import jax.numpy as jnp
from jax import scipy as jscipy


from google.colab import files
uploaded = files.upload()

import pandas as pd
file = next(iter(uploaded))
df = pd.read_csv(file)
print(df)

age = df['age'].values
answer = df['recognized_it'].values
## 1
import numpy as np
from scipy.optimize import minimize_scalar, brent

def func(x):
  return (x-0.3)**2*np.exp(x)

def brent_min(f, x_tol=1e-6):
    res= minimize_scalar(f, method='Brent', tol=x_tol)
    return res.x, res.fun


x_min, y_min = brent_min(func)
x_min1 = brent(func) ## directly used by the package


print(" Brent's 1D method:")
print("Minimum x:", x_min)
print("Minimum y:", y_min)

print("\nScipy's Brent method:")
print("Minimum x:", x_min1)
print("Minimum y:", func(x_min1))

## 2

#L
def logistic_func(params, x):
    beta0, beta1 = params
    return 1 / (1 + jnp.exp(-(beta0 + beta1*x)))

##find log_lilelihood func to make the process of getting maximum value easier.
## we basically find the value at the derivative of log(lL) is 0.
def log_likelihood(params, x, y):  ## x is age, y is answer
    p = logistic_func(params, x)
    p = jnp.clip(p, 1e-15, 1 - 1e-15) # Avoid the log of zero to numerical stability
    return -jnp.sum(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))

## take the gradiant of the log(L)
grad_log_likelihood = grad(log_likelihood)

# take the Hessian of the log(L)
hess_log_likelihood = hessian(log_likelihood)

# Set Initialize parameters to 0
ini_params = jnp.array([0.0, 0.0])

# Use an optimizer to find the maximum likelihood

opt_res = jscipy.optimize.minimize(lambda params: log_likelihood(params, age, answer), ini_params,method='BFGS')

Optbeta0, Optbeta1 = opt_res.x ## optimized parameters


# Calculate the standard errors and covariance matrix
comatrix = jscipy.linalg.inv(hess_log_likelihood(opt_res.x, age, answer))## plug the optimal parameter into log（L）
std_errors = jnp.sqrt(jnp.diag(comatrix))

# Print results
print("Optimized parameters:")
print("Beta0:", Optbeta0)
print("Beta1:", Optbeta1)

print("SE(Beta0):", std_errors[0])
print("SE(Beta1):", std_errors[1])

# Plot the logistic model and the answers
X= jnp.linspace(min(age), max(age), 100)
Y = logistic_function([Optbeta0, Optbeta1], X) ## plug the optimal beta0 and beta1

plt.scatter(age, answer, color='blue')
plt.plot(X, Y, color='red', label='Logistic model')
plt.xlabel('Age')
plt.ylabel('Probability of answering "yes"')
plt.legend()
plt.show()
comatrix
