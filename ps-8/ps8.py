# -*- coding: utf-8 -*-
##Ps8

import matplotlib.pyplot as plt
import numpy as np

## 7.3
from google.colab import files
uploaded = files.upload()

# Get the file name
file_name = list(uploaded.keys())[0]

# Open and read the contents of the uploaded text file
with open(file_name, 'r') as file:
    file_contents = file.read()

# Display the contents of the text file
print(file_contents)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

def plot_waveform(time, signal):
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.title('Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.show()

def plot_fourier_magnitudes(freq, mag, n):
    plt.figure(figsize=(10, 4))
    plt.plot(freq[:n], mag[:n])
    plt.title('Discrete Fourier Transform ')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

def main(file_path):   ## trumpet.txt

    signal = np.loadtxt(file_path)
    time = np.arange(0, len(signal))
    plot_waveform(time, signal)  ## defined before
    freq = np.fft.fftfreq(len(signal))  ## calculate the fourier transform signal
    fft_result = fft(signal)  ## fourier transform of it
    plot_fourier_magnitudes(freq, np.abs(fft_result), 10000)
if __name__ == "__main__":
  main('trumpet.txt')

## 8.3
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def lorenz(t, L, sigma, r, b):
    x,y,z=L
    dx_dt = sigma * (y - x)
    dy_dt = r * x - y - x * z
    dz_dt = x * y - b * z
    return [dx_dt, dy_dt, dz_dt]


sigma = 10
r = 28
b = 8/3


initial_conditions = [0, 1, 0]

t_range = (0, 50)

# use solve_ivp function
solution = solve_ivp(
    lorenz,
    t_range,
    initial_conditions,
    args=(sigma, r, b), ## 10, 38, and 8/3 are plugged into the function
    dense_output=True,
    max_step=0.01
)

# Extract the solution
t = np.linspace(0, 50, 1000) ## from 0 to 50 for t
u = solution.sol(t)

plt.plot(t, u[1], label='y')  ## u[0] = x; u[1]=y; u[2]=z
plt.title('Lorenz Equations Solution')
plt.xlabel('Time')
plt.ylabel('y-coordinate')
plt.legend()
plt.show()

plt.plot(u[0], u[2], label='z vs x')
plt.title('Lorenz Equations Solution')
plt.xlabel('x-coordinate')
plt.ylabel('z-coordinate')
plt.legend()
plt.show()
