import numpy as np
import extract

tpf = extract.TargetPixelFile('ktwo210459199-c04_lpd-targ.fits.gz')
mask = tpf.aperture_mask()
raw = tpf.raw_lightcurve()
xbar, ybar = tpf.centroids()


import matplotlib.pyplot as pl
pl.figure()
pl.subplot(311)
pl.plot(tpf.time, raw / np.median(raw))
pl.subplot(312)
pl.plot(tpf.time, xbar)
pl.subplot(313)
pl.plot(tpf.time, ybar)
pl.savefig('lc.png')
pl.close()

pl.figure()
pl.scatter(xbar, raw / np.median(raw), c='blue')
pl.scatter(ybar, raw / np.median(raw), c='red')

pl.savefig('lc2.png')
pl.close()