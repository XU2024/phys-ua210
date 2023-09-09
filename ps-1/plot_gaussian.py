import numpy as np
import matplotlib.pyplot as plt 
sigma = 3
miu = 0
x = np.linspace(-10, 10)
f = np.exp(-((x-miu)/sigma)**2/2)/(sigma*np.sqrt(2*np.pi))
plt.plot(x, f)
plt.savefig('gaussian.png')
plt.show()








