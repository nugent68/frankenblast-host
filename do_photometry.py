import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import astropy.units as u
from astropy.wcs import WCS
from photutils.aperture import EllipticalAperture, aperture_photometry, SkyEllipticalAperture
from photutils.utils import calc_total_error
from photutils.background import LocalBackground,MeanBackground


from create_apertures import estimate_background

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def do_global_photometry(transient, filter=None, aperture=None, fwhm_correction=True, show_plot=False):
    # Sometimes we can specify a filter / aperture...
    if filter is not None and aperture is not None:
        # Find which filter I am talking about
        good_cutout = None
        for cutout in transient.cutouts:
            if cutout['filter'] == filter:
                good_cutout = cutout
                break

        file_path = good_cutout['file_path']
        # Load the FITS image
        image = fits.open(file_path)
        wcs = WCS(image[0].header)

        if fwhm_correction:
            aperture = adjust_aperture(aperture, cutout)

        if show_plot:
            # Normalize to look pretty
            norm = ImageNormalize(image[0].data, interval=ZScaleInterval())

            # Create the plot
            plt.figure(figsize=(8, 8))
            plt.imshow(image[0].data, cmap='gray', origin='lower', norm=norm)
            pixel_aperture = aperture.to_pixel(wcs)
            pixel_aperture.plot(color='red', lw=2, label="Elliptical Aperture")
            plt.colorbar(label='Pixel Value')
            plt.title(f"Image: {filter.name}")
            plt.xlabel("X Pixels")
            plt.ylabel("Y Pixels")
            plt.show()

        phot = do_aperture_photometry(image, aperture, filter)
        image.close()

    elif filter is not None and aperture == None:
        print('no apr')
        # Find which filter I am talking about
        good_cutout = None
        for cutout in transient.cutouts:
            if cutout['filter'] == filter:
                good_cutout = cutout
                break
        
        file_path = good_cutout['file_path']
        # Load the FITS image
        image = fits.open(file_path)
        wcs = WCS(image[0].header)

        # ANYA - adding the FWHM correction when filter is available, but
        # Get all apertures from filters with aperture sizes recorded
        all_apr = np.array(transient.global_apertures)[np.where(np.array(transient.global_apertures) != None)]        

        # Get all central wavelengths from corresponding filters
        all_wave_eff = np.array([c['filter'].wavelength_eff_angstrom 
                                 for c in transient.cutouts])[np.where(np.array(transient.global_apertures) != None)]

        # Find the nearest filter in wavelength space and use that aperture size
        placeholder_apr = all_apr[find_nearest(all_wave_eff, filter.wavelength_eff_angstrom)]
        
        # Perform FWHM correction
        aperture = adjust_aperture(placeholder_apr, cutout)

        if show_plot:
            # Normalize to look pretty
            norm = ImageNormalize(image[0].data, interval=ZScaleInterval())

            # Create the plot
            plt.figure(figsize=(8, 8))
            plt.imshow(image[0].data, cmap='gray', origin='lower', norm=norm)
            pixel_aperture = aperture.to_pixel(wcs)
            pixel_aperture.plot(color='red', lw=2, label="Elliptical Aperture")
            plt.colorbar(label='Pixel Value')
            plt.title(f"Image: {filter.name}")
            plt.xlabel("X Pixels")
            plt.ylabel("Y Pixels")
            plt.show()

        phot = do_aperture_photometry(image, aperture, filter)
        image.close()

        
    return phot


def do_aperture_photometry(image, sky_aperture, filter):
    """
    Performs Aperture photometry
    """
    image_data = image[0].data
    wcs = WCS(image[0].header)

    # get the background
    try:
        background = estimate_background(image, filter.name)
    except ValueError:
        # indicates poor image data
        return {
            "flux": None,
            "flux_error": None,
            "magnitude": None,
            "magnitude_error": None,
        }

    # is the aperture inside the image?
    bbox = sky_aperture.to_pixel(wcs).bbox
    if (
        bbox.ixmin < 0
        or bbox.iymin < 0
        or bbox.ixmax > image_data.shape[1]
        or bbox.iymax > image_data.shape[0]
    ):
        return {
            "flux": None,
            "flux_error": None,
            "magnitude": None,
            "magnitude_error": None,
        }

    # if the image pixels are all zero, let's assume this is masked
    # even GALEX FUV should have *something*
    phot_table_maskcheck = aperture_photometry(image_data, sky_aperture, wcs=wcs)
    if phot_table_maskcheck["aperture_sum"].value[0] == 0:
        return {
            "flux": None,
            "flux_error": None,
            "magnitude": None,
            "magnitude_error": None,
        }

    background_subtracted_data = image_data - background.background

    # I think we need a local background subtraction for WISE
    # the others haven't given major problems
    if "WISE" in filter.name:
        aper_pix = sky_aperture.to_pixel(wcs)
        lbg = LocalBackground(aper_pix.a, aper_pix.a * 2)
        local_background = lbg(
            background_subtracted_data, aper_pix.positions[0], aper_pix.positions[1]
        )
        background_subtracted_data -= local_background

    if filter.image_pixel_units == "counts/sec":
        error = calc_total_error(
            background_subtracted_data,
            background.background_rms,
            float(image[0].header["EXPTIME"]),
        )

    else:
        error = calc_total_error(
            background_subtracted_data, background.background_rms, 1.0
        )

    phot_table = aperture_photometry(
        background_subtracted_data, sky_aperture, wcs=wcs, error=error
    )
    uncalibrated_flux = phot_table["aperture_sum"].value[0]
    if "2MASS" not in filter.name:
        uncalibrated_flux_err = phot_table["aperture_sum_err"].value[0]
    else:
        # 2MASS is annoying
        # https://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/
        n_pix = (
            np.pi
            * sky_aperture.a.value
            * sky_aperture.b.value
            * filter.pixel_size_arcsec**2.0
        )
        uncalibrated_flux_err = np.sqrt(
            uncalibrated_flux / (10 * 6)
            + 4 * n_pix * 1.7**2.0 * np.median(background.background_rms) ** 2
        )

    # check for correlated errors
    aprad, err_adjust = filter.correlation_model()
    if aprad is not None:
        image_aperture = sky_aperture.to_pixel(wcs)

        err_adjust_interp = np.interp(
            (image_aperture.a + image_aperture.b) / 2.0, aprad, err_adjust
        )
        uncalibrated_flux_err *= err_adjust_interp

    if filter.magnitude_zero_point_keyword is not None:
        zpt = image[0].header[filter.magnitude_zero_point_keyword]
    elif filter.image_pixel_units == "counts/sec":
        zpt = filter.magnitude_zero_point 
    else:
        zpt = filter.magnitude_zero_point +  2.5 * np.log10(image[0].header["EXPTIME"])
    

    flux = flux_to_mJy_flux(uncalibrated_flux, zpt)
    flux = flux*10 ** (-0.4 * filter.ab_offset)
    flux_error = fluxerr_to_mJy_fluxerr(uncalibrated_flux_err, zpt)
    flux_error = flux_error*10 ** (-0.4 * filter.ab_offset)

    magnitude = flux_to_mag(uncalibrated_flux, zpt)
    magnitude_error = fluxerr_to_magerr(uncalibrated_flux, uncalibrated_flux_err)
    if magnitude != magnitude:
        magnitude, magnitude_error = None, None
    if flux != flux or flux_error != flux_error:
        flux, flux_error = None, None

    # wave_eff = filter.transmission_curve().wave_effective
    return {
        "flux": flux,
        "flux_error": flux_error,
        "magnitude": magnitude,
        "magnitude_error": magnitude_error,
    }


def flux_to_mJy_flux(flux, zero_point_mag_in):
    """
    Converts flux of instrument to microjansky (NOT mili)
    """
    return flux * 10 ** (-0.4 * (zero_point_mag_in - 23.9))

def fluxerr_to_mJy_fluxerr(fluxerr, zero_point_mag_in):
    """
    Converts flux to magnitude
    """
    return fluxerr * 10 ** (-0.4 * (zero_point_mag_in - 23.9))

def flux_to_mag(flux, zero_point_mag):
    """
    Converts flux to magnitude
    """
    return -2.5 * np.log10(flux) + zero_point_mag

def fluxerr_to_magerr(flux, fluxerr):
    """
    Converts flux to magnitude
    """
    return 1.0857 * fluxerr / flux


def adjust_aperture(aperture, cutout):
    """
    Adjust the elliptical aperture based on the FWHM of the filter.

    Parameters
    ----------
    aperture : SkyEllipticalAperture
        The original aperture to be adjusted.
    cutout : dict
        Dictionary containing 'file_path' and the `Filter` object for the cutout.

    Returns
    -------
    SkyEllipticalAperture
        Adjusted aperture with corrected semi-major and semi-minor axes.
    """
    # Extract FWHM for the filter in arcseconds
    fwhm = cutout['filter'].image_fwhm_arcsec

    if fwhm is None:
        raise ValueError(f"FWHM for filter {cutout['filter'].name} is not defined.")

    # Correct the semi-major and semi-minor axes
    semi_major_axis = (
        aperture.a + fwhm * u.arcsec
    ).to(u.deg)
    semi_minor_axis = (
        aperture.b + fwhm * u.arcsec
    ).to(u.deg)

    # Return the adjusted aperture
    return SkyEllipticalAperture(
        positions=aperture.positions,
        a=semi_major_axis,
        b=semi_minor_axis,
        theta=aperture.theta
    )


# Work in progress.
# def pick_best_aperture()