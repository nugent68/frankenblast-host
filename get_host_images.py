import settings
import yaml
from collections import namedtuple
import os
import time
from astroquery.hips2fits import hips2fits
from astropy.units import Quantity
import astropy
import astropy.units as u
import numpy as np
import requests
import pandas as pd
from astropy.io import fits
from io import BytesIO
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astroquery.mast import Observations
from astroquery.sdss import SDSS
from astroquery.skyview import SkyView
from pyvo.dal import sia
import re
from io import BytesIO
import astropy.table as at

from classes import Filter, Survey, Transient

DOWNLOAD_MAX_TRIES = 1
DOWNLOAD_SLEEP_TIME = 0




def cutout(transient, survey, fov=Quantity(0.1, unit="deg")):
    """
    Download image cutout data from a survey.
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :survey : :class: Survey
        Named tuple containing metadata for the survey the image is to be
        downloaded from.
    :fov : :class:`~astropy.units.Quantity`,
    default=Quantity(0.2,unit='deg')
        Field of view of the cutout image, angular length of one of the sides
        of the square cutout. Angular astropy quantity. Default is angular
        length of 0.2 degrees.
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
        Image cutout in fits format or if the image cannot be download due to a
        `ReadTimeoutError` None will be returned.
    """
    # need to make sure the cache doesn't overfill
    astropy.utils.data.clear_download_cache()
    num_pixels = int(fov.to(u.arcsec).value / survey.pixel_size_arcsec)

    err = None   # Ensure err is defined in all cases
    status = 1
    n_iter = 0
    while status == 1 and n_iter < DOWNLOAD_MAX_TRIES:
        if survey.image_download_method == "hips":
            try:
                fits = hips_cutout(transient, survey, image_size=num_pixels)
                status = 0
            except Exception as e:
                print(f"Conection timed out, could not download {survey.name} data")
                fits = None
                status = 1
                err = e
        else:
            survey_name, filter = survey.name.split("_")
            try:
                fits = download_function_dict[survey_name](
                    transient, filter=filter, image_size=num_pixels
                )
                status = 0
                err = None
            except Exception as e:
                print(f"Could not download {survey.name} data")
                print(f"exception: {e}")
                fits = None
                status = 1
                err = e
        n_iter += 1
        if status == 1:
            time.sleep(DOWNLOAD_SLEEP_TIME)

    return fits, status, err


def survey_list(survey_metadata_path):
    """
    Build a list of filters directly from a metadata file.

    Parameters
    ----------
    survey_metadata_path : str
        Path to a yaml data file containing filter metadata.

    Returns
    -------
    list
        List of filter names.
    """
    Filter._filters = []  # Clear the global filters list

    with open(survey_metadata_path, "r") as stream:
        survey_metadata = yaml.safe_load(stream)

    filter_names = []
    for filter_name, filter_data in survey_metadata.items():
        # Extract survey name from the hips_id (second to last element of the path)
        hips_id = filter_data.get("hips_id", "")
        survey_name = hips_id.split("/")[-2] if "/" in hips_id else None

        Filter(
            name=filter_name,
            survey=survey_name,  # Extracted survey name
            hips_id=hips_id,
            image_download_method=filter_data.get("image_download_method"),
            pixel_size_arcsec=filter_data.get("pixel_size_arcsec"),
            magnitude_zero_point=filter_data.get("magnitude_zero_point"),
            magnitude_zero_point_keyword=filter_data.get("magnitude_zero_point_keyword"),
            image_pixel_units=filter_data.get("image_pixel_units"),
            image_fwhm_arcsec=filter_data.get("image_fwhm_arcsec"),
            wavelength_eff_angstrom=filter_data.get('wavelength_eff_angstrom'),
            ab_offset=filter_data.get('ab_offset'),
            vega_zero_point_jansky = filter_data.get('vega_zero_point_jansky'),
            sedpy_name = filter_data.get('sedpy_name'),
        )
        filter_names.append(filter_name)

    return filter_names


def download_and_save_cutouts(
    transient,
    filters=None,
    fov=Quantity(0.1, unit="deg"),
    media_root="./cutouts/",
    overwrite=False,
):
    """
    Download all available imaging from a list of filters.

    Parameters
    ----------
    transient : Transient
        Transient object containing target information.
    filters : list[Filter], optional
        List of Filter objects to process. Defaults to all filters in `Filter.all()`.
    fov : Quantity
        Field of view for the cutout image.
    media_root : str
        Root directory for saving cutouts.
    overwrite : bool
        Whether to overwrite existing files.

    Returns
    -------
    str
        Processing status.
    """
    if filters is None:
        filters = Filter.all()  # Default to all filters globally

    if not filters:
        print("No filters available for processing.")
        return "No filters to process."

    for filter in filters:
        print(f"Processing filter: {filter.name} from survey {filter.survey}")

        # Use a default value for survey if it is None
        survey_name = filter.survey if filter.survey else "unknown_survey"
        save_dir = os.path.join(media_root, transient.name, survey_name)
        os.makedirs(save_dir, exist_ok=True)

        path_to_fits = os.path.join(save_dir, f"{filter.name}.fits")
        file_exists = os.path.exists(path_to_fits)

        if not overwrite and file_exists:
            print(f"File already exists and overwrite is False: {path_to_fits}")
            continue

        fits, status, err = cutout(transient.coordinates, filter, fov=fov)
        if fits:
            fits.writeto(path_to_fits, overwrite=True)
            print(f"Saved cutout to: {path_to_fits}")
        elif status == 1:
            print(f"Error downloading cutout for {filter.name}: {err}")
        else:
            print(f"No image found for {filter.name}")

    return "processed"



# Define all my cutout types..
def panstarrs_image_filename(position, image_size=None, filter=None):
    """Query panstarrs service to get a list of image names

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :size : int: cutout image size in pixels.
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :filename: str: file name of the cutout
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = (
        f"{service}?ra={position.ra.degree}&dec={position.dec.degree}"
        f"&size={image_size}&format=fits&filters={filter}"
    )

    ### was having SSL errors with pandas, so let's run it through requests
    ### optionally, can edit to do this in an unsafe way
    r = requests.get(url, stream=True)
    r.raw.decode_content = True
    filename_table = pd.read_csv(r.raw, sep="\s+")["filename"]
    return filename_table[0] if len(filename_table) > 0 else None


def hips_cutout(position, survey, image_size=None):
    """
    Download fits image from hips2fits service.

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size : int: cutout image size in pixels.
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """
    fov = Quantity(survey.pixel_size_arcsec * image_size, unit="arcsec")

    fits_image = hips2fits.query(
        hips=survey.hips_id,
        ra=position.ra,
        dec=position.dec,
        width=image_size,
        height=image_size,
        fov=fov,
        projection="TAN",
        format="fits",
    )

    # if the position is outside of the survey footprint
    if np.all(np.isnan(fits_image[0].data)):
        fits_image = None
    return fits_image


def panstarrs_cutout(position, image_size=None, filter=None):
    """
    Download Panstarrs cutout from their own service

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    filename = panstarrs_image_filename(position, image_size=image_size, filter=filter)
    if filename is not None:
        service = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
        fits_url = (
            f"{service}ra={position.ra.degree}&dec={position.dec.degree}"
            f"&size={image_size}&format=fits&red={filename}"
        )
        try:
            r = requests.get(fits_url, stream=True)
        except Exception as e:
            time.sleep(5)
            r = requests.get(fits_url, stream=True)
        fits_image = fits.open(BytesIO(r.content))

    else:
        fits_image = None

    return fits_image


def galex_cutout(position, image_size=None, filter=None):
    """
    Download GALEX cutout from MAST

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    obs = Observations.query_criteria(
        coordinates=position,
        radius="0.2 deg",
        obs_collection="GALEX",
        filters=filter,
        distance=0
    )
    #obs = obs[
    #    (obs["obs_collection"] == "GALEX")
    #    & (obs["filters"] == filter)
    #    & (obs["distance"] == 0)
    #]

    # avoid masked regions
    center = SkyCoord(obs["s_ra"], obs["s_dec"], unit=u.deg)
    sep = position.separation(center).deg
    obs = obs[sep < 0.55]

    if len(obs) > 1:
        obs = obs[obs["t_exptime"] == max(obs["t_exptime"])]

    if len(obs):
        ### stupid MAST thinks we want the exposure time map

        fits_image = fits.open(
            obs["dataURL"][0]
            .replace("-exp.fits.gz", "-int.fits.gz")
            .replace("-gsp.fits.gz", "-int.fits.gz")
            .replace("-rr.fits.gz", "-int.fits.gz")
            .replace("-cnt.fits.gz", "-int.fits.gz")
            .replace("-fcat.ds9reg", "-int.fits.gz")
            .replace("-xd-mcat.fits.gz", f"-{filter[0].lower()}d-int.fits.gz"),
            cache=None,
        )

        wcs = WCS(fits_image[0].header)
        cutout = Cutout2D(fits_image[0].data, position, image_size, wcs=wcs)
        fits_image[0].data = cutout.data
        fits_image[0].header.update(cutout.wcs.to_header())
        if not np.any(fits_image[0].data):
            fits_image = None
    else:
        fits_image = None

    return fits_image


def WISE_cutout(position, image_size=None, filter=None):
    """
    Download WISE image cutout from IRSA

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    band_to_wavelength = {
        "W1": "3.4e-6",
        "W2": "4.6e-6",
        "W3": "1.2e-5",
        "W4": "2.2e-5",
    }

    url = f"https://irsa.ipac.caltech.edu/SIA?COLLECTION=wise_allwise&POS=circle+{position.ra.deg}+{position.dec.deg}+0.002777&RESPONSEFORMAT=CSV&BAND={band_to_wavelength[filter]}&FORMAT=image/fits"
    r = requests.get(url)
    url = None
    for t in r.text.split(","):
        if t.startswith("https"):
            url = t[:]
            break

    # remove the AWS crap messing up the CSV format
    line_out = ""
    for line in r.text.split("\n"):
        try:
            idx1 = line.index("{")
        except ValueError:
            line_out += line[:] + "\n"
            continue
        idx2 = line.index("}")
        newline = line[0 : idx1 + 1] + line[idx2:] + "\n"
        line_out += newline

    data = at.Table.read(line_out, format="ascii.csv")
    exptime = data["t_exptime"][0]

    if url is not None:
        fits_image = fits.open(url, cache=None)

        wcs = WCS(fits_image[0].header)
        cutout = Cutout2D(fits_image[0].data, position, image_size, wcs=wcs)
        fits_image[0].data = cutout.data
        fits_image[0].header.update(cutout.wcs.to_header())
        fits_image[0].header["EXPTIME"] = exptime

    else:
        fits_image = None

    return fits_image


def DES_cutout(position, image_size=None, filter=None):
    """
    Download DES image cutout from NOIRLab

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia/ls_dr9"
    svc_ls_dr9 = sia.SIAService(DEF_ACCESS_URL)

    imgTable = svc_ls_dr9.search(
        (position.ra.deg, position.dec.deg),
        (image_size / np.cos(position.dec.deg * np.pi / 180), image_size),
        verbosity=2,
    ).to_table()

    valid_urls = []
    for img in imgTable:
        if "-depth-" in img["access_url"] and img["obs_bandpass"].startswith(filter):
            valid_urls += [img["access_url"]]

    if len(valid_urls):
        # we need both the depth and the image
        time.sleep(1)
        try:
            fits_image = fits.open(
                valid_urls[0].replace("-depth-", "-image-"), cache=None
            )
        except Exception as e:
            ### found some bad links...
            return None
        if np.shape(fits_image[0].data)[0] == 1 or np.shape(fits_image[0].data)[1] == 1:
            # no idea what's happening here but this is a mess
            return None

        try:
            depth_image = fits.open(valid_urls[0])
        except Exception as e:
            # wonder if there's some issue with other tasks clearing the cache
            time.sleep(5)
            depth_image = fits.open(valid_urls[0])
        wcs_depth = WCS(depth_image[0].header)
        xc, yc = wcs_depth.wcs_world2pix(position.ra.deg, position.dec.deg, 0)

        # this is ugly - just assuming the exposure time at the
        # location of interest is uniform across the image
        if np.shape(depth_image[0].data) == (1, 1):
            exptime = depth_image[0].data[0][0]
        else:
            exptime = depth_image[0].data[int(yc), int(xc)]
        if exptime == 0:
            fits_image = None
        else:
            wcs = WCS(fits_image[0].header)
            cutout = Cutout2D(fits_image[0].data, position, image_size, wcs=wcs)
            fits_image[0].data = cutout.data
            fits_image[0].header.update(cutout.wcs.to_header())
            fits_image[0].header["EXPTIME"] = exptime
    else:
        fits_image = None

    return fits_image


def TWOMASS_cutout(position, image_size=None, filter=None):
    """
    Download 2MASS image cutout from IRSA

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    irsaquery = f"https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia?POS={position.ra.deg},{position.dec.deg}&SIZE=0.01"
    response = requests.get(url=irsaquery)

    fits_image = None
    for line in response.content.decode("utf-8").split("<TD><![CDATA["):
        if re.match(f"https://irsa.*{filter.lower()}i.*fits", line.split("]]>")[0]):
            fitsurl = line.split("]]")[0]

            fits_image = fits.open(fitsurl, cache=None)
            wcs = WCS(fits_image[0].header)

            if position.contained_by(wcs):
                break

    if fits_image is not None:
        cutout = Cutout2D(fits_image[0].data, position, image_size, wcs=wcs)
        fits_image[0].data = cutout.data
        fits_image[0].header.update(cutout.wcs.to_header())

    else:
        fits_image = None

    return fits_image


def SDSS_cutout(position, image_size=None, filter=None):
    """
    Download SDSS image cutout from astroquery

    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    sdss_baseurl = "https://data.sdss.org/sas"
    print(position)

    xid = SDSS.query_region(position, radius=0.05 * u.deg)
    if xid is None or len(xid) == 0:
        return None
    
    image_pos = SkyCoord(xid['ra'],xid['dec'],unit=u.deg)
    sep = position.separation(image_pos)
    iSep = np.where(sep == np.min(sep))[0]
    

    # old (better, but deprecated) version
    #url = f"https://dr12.sdss.org/fields/raDec?ra={position.ra.deg}&dec={position.dec.deg}"
    #print(url)
    #rt = requests.get(url)



    regex = "<dt>run<\/dt>.*<dd>.*<\/dd>"
    run = xid['run'][iSep][0] #re.findall("<dt>run</dt>\n.*<dd>([0-9]+)</dd>", rt.text)[0]
    rerun = xid['rerun'][iSep][0] #re.findall("<dt>rerun</dt>\n.*<dd>([0-9]+)</dd>", rt.text)[0]
    camcol = xid['camcol'][iSep][0] #re.findall("<dt>camcol</dt>\n.*<dd>([0-9]+)</dd>", rt.text)[0]
    field = xid['field'][iSep][0] #re.findall("<dt>field</dt>\n.*<dd>([0-9]+)</dd>", rt.text)[0]

    # a little latency so that we don't look like a bot to SDSS?
    time.sleep(1)
    link = SDSS.IMAGING_URL_SUFFIX.format(
        base=sdss_baseurl,
        run=int(run),
        dr=16,
        instrument="eboss",
        rerun=int(rerun),
        camcol=int(camcol),
        field=int(field),
        band=filter,
    )

    fits_image = fits.open(link, cache=None)

    wcs = WCS(fits_image[0].header)
    cutout = Cutout2D(fits_image[0].data, position, image_size, wcs=wcs)
    fits_image[0].data = cutout.data
    fits_image[0].header.update(cutout.wcs.to_header())

    # else:
    #    fits_image = None

    return fits_image


import os

def get_cutouts(transient, base_dir="./cutouts/"):
    """
    Get a list of available cutouts for a transient.

    Parameters
    ----------
    transient_name : str
        The name of the transient (e.g., "AT2024abyq").
    base_dir : str, optional
        Base directory where cutouts are stored, default is "./cutouts/".

    Returns
    -------
    list of dict
        A list of dictionaries, each containing the file path and associated Filter object.
        Example: [{'file_path': 'path/to/file.fits', 'filter': <Filter object>}]
    """
    transient_name = transient.name
    cutouts_dir = os.path.join(base_dir, transient_name)
    if not os.path.exists(cutouts_dir):
        print(f"No cutouts directory found for transient {transient_name}")
        return []

    filters = Filter.all()  # Get all filter instances
    available_cutouts = []

    for root, _, files in os.walk(cutouts_dir):
        for file in files:
            if file.endswith(".fits"):
                file_path = os.path.join(root, file)
                # Extract filter name from the file name
                filter_name = os.path.splitext(file)[0]  # Remove ".fits"
                # Match the filter with the corresponding Filter object
                filter_obj = next((f for f in filters if f.name == filter_name), None)
                if filter_obj:
                    available_cutouts.append({"file_path": file_path, "filter": filter_obj})

    return available_cutouts


download_function_dict = {
    "PanSTARRS": panstarrs_cutout,
    "GALEX": galex_cutout,
    "2MASS": TWOMASS_cutout,
    "WISE": WISE_cutout,
    "DES": DES_cutout,
    "SDSS": SDSS_cutout,
}
