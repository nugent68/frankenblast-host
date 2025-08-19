import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
import astropy.units as u
from photutils.background import SExtractorBackground, Background2D, MeanBackground
from photutils.segmentation import detect_sources, SourceCatalog
from photutils.aperture import EllipticalAperture




def estimate_background(image, filter_name=None):
    """
    Estimates the background of an image
    Parameters
    ----------
    :image : :class:`~astropy.io.fits.HDUList`
        Image to have the background estimated of.
    Returns
    -------
    :background : :class:`~photutils.background.Background2D`
        Background estimate of the image
    """
    image_data = image[0].data
    box_size = int(0.1 * np.sqrt(image_data.size))

    # GALEX needs mean, not median - median just always comes up with zero
    if filter_name is not None and "GALEX" in filter_name:
        bkg = MeanBackground(SigmaClip(sigma=3.0))
    else:
        bkg = SExtractorBackground(sigma_clip=None)

    try:
        return Background2D(image_data, box_size=box_size, bkg_estimator=bkg)
    except ValueError:
        return Background2D(
            image_data, box_size=box_size, exclude_percentile=50, bkg_estimator=bkg
        )

def construct_aperture(cutout, position, threshold_sigma=None):
    """
    Construct an elliptical aperture at the position in the image
    Parameters
    ----------
    :image : :class:`~astropy.io.fits.HDUList`
    Returns
    -------
    """


    # Extract the file path from the dictionary
    file_path = cutout['file_path']

    # Load the FITS image
    image = fits.open(file_path)
    wcs = WCS(image[0].header)
    background = estimate_background(image)

    # found an edge case where deblending isn't working how I'd like it to
    # so if it's not finding the host, play with the default threshold
    def get_source_data(threshhold_sigma):
        catalog = build_source_catalog(
            image, background, threshhold_sigma=threshhold_sigma
        )
        if catalog is None:
            return None, 100
        source_data = match_source(position, catalog, wcs)

        source_ra, source_dec = wcs.wcs_pix2world(
            source_data.xcentroid, source_data.ycentroid, 0
        )
        source_position = SkyCoord(source_ra, source_dec, unit=u.deg)
        source_separation_arcsec = position.separation(source_position).arcsec
        return source_data, source_separation_arcsec

    if threshold_sigma is None:
        iter = 0
        source_separation_arcsec = 100
        while source_separation_arcsec > 5 and iter < 5:
            source_data, source_separation_arcsec = get_source_data(5 * (iter + 1))
            iter += 1
        # look for sub-threshold sources
        # if we still can't find the host
        if source_separation_arcsec > 5:
            source_data, source_separation_arcsec = get_source_data(2)

        # make sure we know this failed
        if source_separation_arcsec > 5:
            return None
    else:
        source_data, source_separation_arcsec = get_source_data(threshold_sigma)
        # make sure we know this failed
        if source_separation_arcsec > 5:
            return None
    image.close()
    return elliptical_sky_aperture(source_data, wcs)

def build_source_catalog(image, background, threshhold_sigma=3.0, npixels=10):
    """
    Constructs a source catalog given an image and background estimation
    Parameters
    ----------
    :image :  :class:`~astropy.io.fits.HDUList`
        Fits image to construct source catalog from.
    :background : :class:`~photutils.background.Background2D`
        Estimate of the background in the image.
    :threshold_sigma : float default=2.0
        Threshold sigma above the baseline that a source has to be to be
        detected.
    :n_pixels : int default=10
        The length of the size of the box in pixels used to perform segmentation
        and de-blending of the image.
    Returns
    -------
    :source_catalog : :class:`photutils.segmentation.SourceCatalog`
        Catalog of sources constructed from the image.
    """

    image_data = image[0].data
    background_subtracted_data = image_data - background.background
    threshold = threshhold_sigma * background.background_rms

    segmentation = detect_sources(
        background_subtracted_data, threshold, npixels=npixels
    )
    if segmentation is None:
        return None
    # deblended_segmentation = deblend_sources(
    #     background_subtracted_data, segmentation, npixels=npixels
    # )
    return SourceCatalog(background_subtracted_data, segmentation)

def match_source(position, source_catalog, wcs):
    """
    Match the source in the source catalog to the host position
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        On Sky position of the source to be matched.
    :source_catalog : :class:`~photutils.segmentation.SourceCatalog`
        Catalog of sources.
    :wcs : :class:`~astropy.wcs.WCS`
        World coordinate system to match the sky position to the
        source catalog.
    Returns
    -------
    :source : :class:`~photutils.segmentation.SourceCatalog`
        Catalog containing the one matched source.
    """

    host_x_pixel, host_y_pixel = wcs.world_to_pixel(position)
    source_x_pixels, source_y_pixels = (
        source_catalog.xcentroid,
        source_catalog.ycentroid,
    )
    closest_source_index = np.argmin(
        np.hypot(host_x_pixel - source_x_pixels, host_y_pixel - source_y_pixels)
    )

    return source_catalog[closest_source_index]

def elliptical_sky_aperture(source_catalog, wcs, aperture_scale=3.0):
    """
    Constructs an elliptical sky aperture from a source catalog
    Parameters
    ----------
    :source_catalog: :class:`~photutils.segmentation.SourceCatalog`
        Catalog containing the source to get aperture information from.
    :wcs : :class:`~astropy.wcs.WCS`
        World coordinate system of the source catalog.
    :aperture_scale: float default=3.0
        Scale factor to increase the size of the aperture
    Returns
    -------
    :sky_aperture: :class:`~photutils.aperture.SkyEllipticalAperture`
        Elliptical sky aperture of the source in the source catalog.
    """
    center = (source_catalog.xcentroid, source_catalog.ycentroid)
    semi_major_axis = source_catalog.semimajor_sigma.value * aperture_scale
    semi_minor_axis = source_catalog.semiminor_sigma.value * aperture_scale
    orientation_angle = source_catalog.orientation.to(u.rad).value
    pixel_aperture = EllipticalAperture(
        center, semi_major_axis, semi_minor_axis, theta=orientation_angle
    )
    pixel_aperture = source_catalog.kron_aperture
    return pixel_aperture.to_sky(wcs)