'''7. Given signals x = [1, 2, 3, 4] and y = [4, 3, 2, 1], calculate the autocorrelation of x, the cross-correlation of x and y, and demonstrate the convolution of x and y in 1D.'''

import numpy as np
import matplotlib.pyplot as plt

# Given signals
x = np.array([1, 2, 3, 4])
y = np.array([4, 3, 2, 1])

# Autocorrelation of x
autocorr_x = np.correlate(x, x, mode='full')

# Cross-correlation of x and y
crosscorr_xy = np.correlate(x, y, mode='full')

# Convolution of x and y
convolution_xy = np.convolve(x, y, mode='full')

# Display the results
print("Autocorrelation of x:", autocorr_x)
print("Cross-correlation of x and y:", crosscorr_xy)
print("Convolution of x and y:", convolution_xy)

# Plotting the results
plt.figure(figsize=(12, 4))

# Autocorrelation plot
plt.subplot(1, 3, 1)
plt.stem(autocorr_x)
plt.title('Autocorrelation of x')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')

# Cross-correlation plot
plt.subplot(1, 3, 2)
plt.stem(crosscorr_xy)
plt.title('Cross-correlation of x and y')
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')

# Convolution plot
plt.subplot(1, 3, 3)
plt.stem(convolution_xy)
plt.title('Convolution of x and y')
plt.xlabel('Index')
plt.ylabel('Convolution')

plt.tight_layout()
plt.show()
