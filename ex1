from astropy.io import fits
import math
import numpy as np
import matplotlib.pyplot as plt


img = fits.open("data.fits")  
img_data = [i.data for i in img[1:]] # data lists
mean_img = np.mean(img_data, axis=0)
median_img = np.median(img_data, axis=0)

#slide 26
res_mean = []
res_med = []
for yy in range((mean_img.shape[1] - 1) * 3):
	x = mean_img.shape[0] / 2
	y = yy / 3
	i = math.floor(x)
	j = math.floor(y)
	fx = x - i
	fy = y - j
	i_res_mean = (1 - fx) * (1 - fy) * mean_img[i, j] + fy * (1 - fx) * mean_img[i, j + 1] + fx * (1 - fy) * mean_img[i + 1, j] + fx * fy * mean_img[i + 1, j + 1]
	res_mean.append(i_res_mean)
	i_res_med = (1 - fx) * (1 - fy) * median_img[i, j] + fy * (1 - fx) * median_img[i, j + 1] + fx * (1 - fy) * median_img[i + 1, j] + fx * fy * median_img[i + 1, j + 1]
	res_med.append(i_res_med)

fig, axs = plt.subplots(2, 2)
axs[0][0].imshow(median_img , cmap= 'gist_heat')
axs[0][1].imshow(mean_img, cmap= 'gist_heat')
axs[0][0].set_title('Mean')
axs[0][1].set_title('Median')
axs[1][0].plot(range((mean_img.shape[1] - 1) * 3), res_mean)
axs[1][1].plot(range((mean_img.shape[1] - 1) * 3), res_med)
fig.set_size_inches(20, 10)
plt.show()
