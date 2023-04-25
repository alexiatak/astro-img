
import numpy
from scipy import fft
from astropy.io import fits
from matplotlib import pyplot as plt

img = fits.open("data.fits")  
fourier_img = fft.fft2(img[1].data)
qualities = []

starting = 1
with_step = 1
finishing = 98 + with_step

for p in range(starting, finishing, with_step):
    q = numpy.percentile(numpy.abs(fourier_img), 100 - p)
    c_img = numpy.where(numpy.abs(fourier_img) > q, fourier_img, 0)
    compressed_img=numpy.abs(fft.ifft2(c_img))
    ql=(numpy.sum(numpy.where(img[1].data > 0, (img[1].data - compressed_img) ** 2 / img[1].data, 0)))
    qualities.append(numpy.mean(ql))
      
plt.xlabel("оставленные коэффициенты Фурье., %")
plt.ylabel("качество")
plt.scatter(range(starting,finishing, with_step), qualities,color='pink')
fig = plt.gcf()
fig.savefig("result.png")
plt.show()
