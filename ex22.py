import numpy
from scipy import fft
from astropy.io import fits


ffile = fits.open('noised.fits')
img = ffile[0].data
transf_img= fft.fft2(img)


#d1 = (804, 881)
#d2 = (834, 911)
#transf_img[d1] = numpy.median(transf_img[d1[0] - 1:d1[0] + 1, d1[1] - 1:d1[1] + 1])
#transf_img[d2] = numpy.median(transf_img[d2[0] - 1:d2[0] + 1, d2[1] - 1:d2[1] + 1])


img = numpy.abs(fft.ifft2(transf_img))
hdu = fits.PrimaryHDU(data=img)
fits.HDUList([hdu]).writeto("result.fits", overwrite=True)
