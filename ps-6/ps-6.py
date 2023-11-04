

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np


hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

# a)
for i in range(7):  ##0-6
    plt.plot(logwave, flux[i],label=f'Galaxy {i+ 1}') # use f to make number valid
    plt.xlabel('log10(Wavelength )')
    plt.ylabel('Flux')
    plt.title(f'Spectrum of galaxy{i+1}')
    plt.legend()
    plt.show()

# b) nomralization: flux / sum(flux)
sum_flux = np.sum(flux, axis=1)  #
norm_flux = flux / sum_flux[:,np.newaxis]## transfer to 2D column
 
# c) mean flux at each wavelength
mean_flux = np.mean(norm_flux, axis=0)

residual = norm_flux - mean_flux

print('The normalized fluxes are', norm_flux)
print('The residues are', residual )


#d)

N_gal, N_wave = residual.shape ## row number is #galaxies
C = (1/N_gal) * residual.T @ residual

eigenvalues, eigenvectors = np.linalg.eigh(C)

# Sort the eigenvectors by eigenvalues in descending order
s = np.argsort(eigenvalues)[::-1] # sorted indices
eigenvalues = eigenvalues[s] # sort eigenvalue in descending order
eigenvectors = eigenvectors[:, s] # select eigenvector according to the sorted eigenvalue

for i in range(5):
    plt.figure()
    plt.plot(eigenvectors[:, i ])## plot according to the index
    plt.title(f'Eigenvector {i + 1}')
    plt.xlabel('Wavelength')
    plt.grid(True)

plt.show()

#e)
U, S, VT = np.linalg.svd(residual, full_matrices=False)

eigenvectors_svd = VT.T ## take eigenvector directly from VT matrix

for i in range(5):
    plt.figure()
    plt.plot(eigenvectors_svd[:, i])
    plt.title(f'Eigenvector SVD {i + 1}')
    plt.xlabel('Wavelength Index')
    plt.grid(True)

plt.show()

#g

Nc = 20


coefficients = np.dot(residual, eigenvectors[:, :Nc])


spectra = np.dot(coefficients, eigenvectors[:, :Nc].T)


spectra += mean_spectrum 

# Extract the coefficients for c0, c1, and c2
c0 = coefficients[:, 0]
c1 = coefficients[:, 1]
c2 = coefficients[:, 2]

#h
# Plot c0 vs c1
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(c0, c1, marker='o', color='blue')
plt.xlabel('c0')
plt.ylabel('c1')
plt.title('c0 vs c1')

# Plot c0 vs c2
plt.subplot(1, 2, 2)
plt.scatter(c0, c2, marker='o', color='red')
plt.xlabel('c0')
plt.ylabel('c2')
plt.title('c0 vs c2')

plt.tight_layout() 
plt.show()

#i 
Nc_values = list(range(1, 21)) 
squared_residuals = np.array([])

for Nc in Nc_values: # for each Nc, we calcualte the multiple squared residues.
    squared_residual = np.mean(((spectra - residual) / residual) ** 2, axis=1)
    squared_residual = np.append(np.mean(squared_residual))


plt.plot(Nc_values, squared_residual, marker='o') plot a graph
plt.xlabel('Nc')
plt.ylabel('Squared Fractional Residuals')
plt.title('Squared Fractional Residuals vs Nc')
plt.grid(True)
plt.show()



# In[ ]:


condition_number = largest_eigenvalue / smallest_eigenvalue

