"""Turn a TPF file into a lightcurve.

Design
~~~~~~
# Objects
- TargetPixelFile(tpf)
- FluxTimeSeries(time, flux)
- CentroidTimeSeries(time, x, y)
- TemperatureTimeSeries(time, teff)
- SystematicsRemover()

# Draft API
TargetPixelFile(tpf).raw_lightcurve() -> FluxTimeSeries
TargetPixelFile(tpf).centroid() -> CentroidTimeSeries
FluxTimeSeries.detrend(timescale) -> FluxTimeSeries  # remove low-frequency signals
FluxTimeSeries.normalize() -> FluxTimeSeries  # continuum-normalize
CentroidTimeSeries(x, y).compute_arclength(reject_outliers=True) -> arclength array
FluxTimeSeries.remove_motion_noise(CentroidTimeSeries) -> FluxTimeSeries  ?

# Example use:
tpf = TargetPixelFile(path)
raw_flux = tpf.raw_lightcurve()  # FluxTimeSeries
low_frequency_component = raw_flux.low_frequency_component()  # FluxTimeSeries
detrended_flux = raw_flux - low_frequency_component  # FluxTimeSeries
arclength = tpf.centroid().arclength(reject_outliers=True)
model = SystematicsModel()
model.fit(arclength, flux)
corrected_detrended_flux = detrended_flux - model.predict(arclength)
calibrated_flux = corrected_detrended_flux + low_frequency_component


TODO
~~~~
* split aperture_mask into "find central star"; allow user to give central coordinate
* dont do the quality cuts early on
* estimate background using median of pixels outside of aperture
* divide lightcurve by its median brightness to continuum-normalize
* enable using different photutils centroid functions
* produce 2D plot showing x vs y
* use multiple targets to compute centroids more accurately
* centroid all stars in K2 and provide as catalog = input to detrending
* have different "detrending" classes; solves problem of fitting function (x,y,..) => correction
* review Erik's approach: https://github.com/petigura/k2phot/blob/master/k2phot/pixdecor.py

* Does the C0 data still contain zeros instead of NaNs?
* When creating pixel mask, should we background-subtract the frames prior to making the median image?
"""
from astropy.io import fits
from astropy.stats.funcs import median_absolute_deviation
import numpy as np
import scipy.ndimage
from photutils.centroids import centroid_com


class TargetPixelFile(object):
    """Enables extraction of raw lightcurves and centroid positions."""

    def __init__(self, path, max_quality=1):
        self.path = path
        self.max_quality = max_quality
        self.hdu = fits.open(path)

    def good_quality_mask(self, max_quality=None):
        """Returns a boolean mask flagging the good-quality cadences."""
        if max_quality is None:
            max_quality = self.max_quality
        return self.hdu[1].data['QUALITY'] < max_quality

    @property
    def n_cadences(self):
        """Returns the number of good-quality cadences."""
        return self.good_quality_mask().sum()

    @property
    def time(self):
        """Returns the time for all good-quality cadences."""
        return self.hdu[1].data['TIME'][self.good_quality_mask()]

    @property
    def flux(self):
        """Returns the flux for all good-quality cadences."""
        return self.hdu[1].data['FLUX'][self.good_quality_mask(), :, :]

    def median_flux(self):
        """Returns the median flux across all cadences."""
        return np.nanmedian(self.flux, axis=0)

    def aperture_mask(self, snr_threshold=5):
        """Returns an aperture photometry mask.

        snr_threshold : float
            Background detection threshold.
        """
        # Find the pixels that are above the threshold in the median flux image
        median = self.median_flux()
        mad = median_absolute_deviation(median[np.isfinite(median)])
        mad_cut = 1.4826 * mad * snr_threshold  # 1.4826 turns MAD into STDEV for a Gaussian

        region = np.where(median > mad_cut, 1, 0)
        # Label all contiguous regions above the threshold
        labels = scipy.ndimage.label(region)[0]
        # Central pixel coordinate
        centralpix = [1 + median.shape[0] // 2,
                      1 + median.shape[1] // 2]  # "//" is the integer division

        # find brightest pix within margin of central pix
        margin = 4
        central_img = median[centralpix[0] - margin: centralpix[0] + margin,
                             centralpix[1] - margin: centralpix[1] + margin]
        # unravel_index converts indices into a tuple of coordinate arrays
        brightestpix = np.unravel_index(central_img.argmax(), central_img.shape)
        bpixy, bpixx = brightestpix

        # Which label corresponds to the brightest pixel?
        regnum = labels[centralpix[0] - 4 + bpixy, centralpix[1] - 4 + bpixx]
        if regnum == 0:  # No pixels above threshold?
            print('WARNING, no star was found in light curve, \
                   {} light curve will be junk!'.format(fn))

        aperture_mask = labels == regnum
        return aperture_mask

    def centroids(self, aperture_mask=None):
        if aperture_mask is None:
            aperture_mask = self.aperture_mask()

        # Initialize return values
        xbar = np.zeros(self.n_cadences)
        ybar = np.zeros(self.n_cadences)

        flux = self.flux
        for i in range(self.n_cadences):
            xbar[i], ybar[i] = centroid_com(flux[i], mask=~aperture_mask)
        return xbar, ybar

    def raw_lightcurve(self, aperture_mask=None):
        if aperture_mask is None:
            aperture_mask = self.aperture_mask()

        lightcurve = np.zeros(self.n_cadences)
        flux = self.flux
        for i in range(self.n_cadences):
            lightcurve[i] = flux[i][aperture_mask].sum()
        return lightcurve


class Lightcurve():

    def __init__(self, time, flux):
        self.time = time
        self.flux = flux

    def detrend(self):
        """
        Steps:
        - Fit low-order polynomial to centroid data to reject outliers
        - Compute arclength from centroid data (using eigenvectors etc)
        """

class Detrender():
    pass

    def detrend(flux):
        pass

class ArclengthDetrender():
    pass




def median_subtract(fla):
    """
    subtract the background from a series of images
    by assuming the aperture is large enough to be
    predominantly background
    """
    for i in range(np.shape(fla)[0]):
        fla[i, :, :] = fla[i, :, :] - np.nanmedian(fla[i, :, :])
    return fla


def extract_lightcurve(fn, qual_cut=False, toss_resat=True,
                       bg_cut=5, skip=None):
    if skip is None:
        skip = 0

    # Read the data into time, fluxarr, and quality
    with fits.open(fn) as f:
        time = f[1].data['TIME'][skip:] - f[1].data['TIME'][0]
        fluxarr = f[1].data['FLUX'][skip:]
        quality = f[1].data['QUALITY'][skip:]

    # Remove data that does not meet the quality criteria
    if qual_cut:
        time = time[quality == 0]
        fluxarr = fluxarr[quality == 0, :, :]
    elif toss_resat:
        # data the cadences where there is a wheel
        # resetuation event
        time = time[quality != 32800]
        fluxarr = fluxarr[quality != 32800, :, :]

    # fix dodgy data: the C0 data release included zeros
    # this will be changed later but we need this
    # fix for now
    fluxarr[fluxarr == 0] = np.nan

    # subtract background
    flux_b = median_subtract(fluxarr)

    # create a median image to calculate where
    # the pixels to use are
    flatim = np.nanmedian(flux_b, axis=0)

    # find pixels that are X MAD above the median
    vals = flatim[np.isfinite(flatim)].flatten()
    # 1.4826 turns a MAD into a STDEV for a Gaussian
    mad_cut = 1.4826 * median_absolute_deviation(vals) * bg_cut

    region = np.where(flatim > mad_cut, 1, 0)
    lab = scipy.ndimage.label(region)[0]

    # find the central pixel ("//" is the integer division)
    imshape = np.shape(flatim)
    centralpix = [1 + imshape[0] // 2, 1 + imshape[1] // 2]

    # find brightest pix within 9x9 of central pix
    centflatim = flatim[centralpix[0] - 4: centralpix[0] + 4,
                        centralpix[1] - 4: centralpix[1] + 4]
    flatimfix = np.where(np.isfinite(centflatim), centflatim, 0)
    # unravel_index converts indices into a tuple of coordinate arrays
    brightestpix = np.unravel_index(flatimfix.argmax(), centflatim.shape)
    bpixy, bpixx = brightestpix

    regnum = lab[centralpix[0] - 4 + bpixy, centralpix[1] - 4 + bpixx]
    if regnum == 0:
        print('WARNING, no star was found in light curve, \
               {} light curve will be junk!'.format(fn))

    # Initialize return values
    lc = np.zeros_like(time)
    xbar = np.zeros_like(time)
    ybar = np.zeros_like(time)

    # there is a loop that performs the aperture photometry
    # lets also calcualte the moments of the image

    # make a rectangular aperture for the moments thing
    ymin = np.min(np.where(lab == regnum)[0])
    ymax = np.max(np.where(lab == regnum)[0])
    xmin = np.min(np.where(lab == regnum)[1])
    xmax = np.max(np.where(lab == regnum)[1])

    momlims = [ymin, ymax + 1, xmin, xmax + 1]

    for i, fl in enumerate(fluxarr):
        lc[i] = np.sum(fl[lab == regnum])
        momim = fl[momlims[0]:momlims[1],
                   momlims[2]:momlims[3]]
        momim[~np.isfinite(momim)] == 0.0
        xbar[i], ybar[i], cov = intertial_axis(momim)

    return (time, lc, xbar - np.mean(xbar), ybar - np.mean(ybar), regnum)


if __name__ == '__main__':
    input_fn = 'ktwo210459199-c04_lpd-targ.fits.gz'
    time, lc, xbar, ybar, regnum = extract_lightcurve(input_fn)

    import pandas as pd
    df = pd.Series(data=lc, index=time)
    """
    import matplotlib.pyplot as pl
    pl.figure()
    pl.scatter()
    """