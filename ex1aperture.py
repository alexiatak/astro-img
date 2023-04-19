from astropy.io import fits
import math
import numpy as np
import matplotlib.pyplot as plt
from photutils import aperture

img = fits.open("data.fits")  
img_data = [i.data for i in img[1:]] # data lists
mean_img = np.mean(img_data, axis=0)
median_img = np.median(img_data, axis=0)

apperture_mean = []
apperture_med = []
for i in range(mean_img.shape[1]-1):
	apperture_mean.append(float(aperture.aperture_photometry(mean_img,aperture.CircularAperture((mean_img.shape[0]/2,mean_img.shape[0]/2), r=i+1))['aperture_sum'][0]))
	apperture_med.append(float(aperture.aperture_photometry(median_img,aperture.CircularAperture((median_img.shape[0]/2,median_img.shape[0]/2), r=i+1))['aperture_sum'][0]))

fig, axs = plt.subplots(2, 2)
axs[0][0].imshow(median_img , cmap= 'gist_heat')
axs[0][1].imshow(mean_img, cmap= 'gist_heat')
axs[0][0].set_title('Mean')
axs[0][1].set_title('Median')
axs[1][0].plot(range(mean_img.shape[1]-1), apperture_mean)
axs[1][1].plot(range(mean_img.shape[1]-1), apperture_med)
fig.set_size_inches(20, 10)
plt.show()
