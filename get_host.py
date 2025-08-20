import os
from datetime import datetime
import pandas as pd
from scipy.stats import gamma, halfnorm, uniform
from astropy.cosmology import WMAP9 as cosmo
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.cosmology import WMAP9 as cosmo
from astropy.coordinates import SkyCoord
import astropy.units as u
from collections import OrderedDict
from classes import Transient as BlastTransient
from classes import Filter, Host
from mwebv_host import get_mwebv
from get_host_images import download_and_save_cutouts, survey_list, get_cutouts
from create_apertures import construct_aperture
from do_photometry import do_global_photometry
from colorama import Fore, Style
import csv
import shutil
import glob

from astro_prost.diagnose import plot_match
from astro_prost.helpers import GalaxyCatalog, Transient, setup_logger, sanitize_input
from astro_prost.associate import log_host_properties, chunks, get_catalogs

import pathlib
from mpire import WorkerPool
import time
from urllib.error import HTTPError
import astropy.units as u
import numpy as np
import requests
import importlib.resources as pkg_resources
import importlib
import warnings

NPROCESS_MAX = np.maximum(os.cpu_count() - 4, 1)

MAX_RETRIES = 3


PROST_ROOT = os.environ.get("PROST_PATH")


DEFAULT_RELEASES = {
    "glade": "latest",
    "decals": "dr9",
    "panstarrs": "dr2",
    "skymapper": "dr4",
    "rubin": "dp0.2"
}


ONLY_OFFSET_CATS = {"skymapper", "rubin"}

# Filter unnecessary warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")


def ProstSingleTransient(
    SN_obj,
    glade_catalog,
    n_samples,
    priors,
    likes,
    cosmo,
    catalogs,
    verbose=0,
    strict_checking=False,
    warn_on_fallback=True,
    plot_match=False,
):
    """Associates a transient with its most likely host galaxy.

    Parameters
    ----------
    SN_obj: FrankenBlast transient class instantiation
    glade_catalog : pandas.DataFrame
        GLADE catalog of galaxies, with sizes and photo-zs.
    n_samples : int
        Number of samples for the Monte Carlo sampling of associations.
    priors : dict
        Dictionary of priors for the run (at least one of redshift, offset, absolute magnitude).!
    likes : dict
        Dictionary of likelihoods for the run (at least one of offset, absolute magnitude).
    cosmo : astropy.cosmology
        Assumed cosmology for the run (defaults to LambdaCDM if unspecified).
    catalogs : dict
        Dict of source catalogs to query, with required key "name" and optional key "release".
    log_fn : str, optional
        The fn associated with the logger.Logger object.
    verbose : int
        The verbosity level of the output.
    strict_checking : boolean, optional
        If true, raises error if catalog doesn't support conditioning on a property requested.
    warn_on_fallback : boolean, optional
        If true, raises warning if catalog doesn't support conditioning on a property requested. 
    plot_match : boolean, optional
        If true, attempts to generate a plot image.

    Returns
    -------
    tuple
        Properties of the first and second-best host galaxy matches, and
        a dictionary of catalog columns (empty if cat_cols=False)

    """

    logger = setup_logger()
    calc_host_props = list(priors.keys())
    condition_host_props = list(priors.keys())
    unsupported_props = {"redshift", "absmag"}.intersection(priors)
    unsupported_catalogs  = ONLY_OFFSET_CATS.intersection(catalogs)

    if unsupported_props and unsupported_catalogs:
        msg = (
            f"{', '.join(sorted(unsupported_catalogs))} "
            f"{'does not support conditioning on' if len(unsupported_catalogs)==1 else 'do not support conditioning on'} "
            f"{', '.join(sorted(unsupported_props))}; falling back to 'offset' only for this subset."
        )

        if strict_checking:
            raise ValueError(
                msg + "\n\nInterested in contributing a photo-z estimator? "
                      "Open an issue at https://github.com/alexandergagliano/Prost/issues."
            )

        if warn_on_fallback:
            logger.warning(msg)


    try:
        transient = Transient(
            name=SN_obj.name,
            position=SN_obj.coordinates,
            redshift=SN_obj.transient_redshift,
            n_samples=n_samples,
            logger=logger
        )
    except (KeyError, TypeError, AttributeError):
        transient = Transient(
            name=SN_obj.name,
            position=SN_obj.coordinates,
            n_samples=n_samples,
            logger=logger
        )

    logger.info(
        f"\n\nAssociating {transient.name} at RA, DEC = "
        f"{transient.position.ra.deg:.6f}, {transient.position.dec.deg:.6f} (redshift {transient.redshift:.3f})"
    )

    for key, val in priors.items():
        transient.set_prior(key, val)

    for key, val in likes.items():
        transient.set_likelihood(key, val)

    if 'redshift' in priors.keys():
        transient.gen_z_samples(n_samples=n_samples)

    # Define result fields and initialize all values
    result = {
        "name": transient.name,
        "host_ra": None,
        "host_dec": None,
        "host_redshift_mean": np.nan,
        "host_redshift_std": np.nan,
        "host_prob": np.nan,
        "smallcone_posterior": np.nan,
        "missedcat_posterior": np.nan,
        "best_cat": np.nan,
    }

    # Define the fields that we extract for best and second-best hosts
    catalog_dict = OrderedDict(get_catalogs(catalogs))

    for cat_name, cat_release in catalog_dict.items():
        if cat_name in ONLY_OFFSET_CATS:
            condition_host_props_cat = ['offset']
        else:
            condition_host_props_cat = condition_host_props

        cat = GalaxyCatalog(name=cat_name, n_samples=n_samples, data=glade_catalog, release=cat_release)

        try:
            cat.get_candidates(transient, time_query=True, logger=logger, cosmo=cosmo)
        except requests.exceptions.HTTPError:
            logger.warning(f"Candidate retrieval failed for {transient.name} in catalog {cat_name} due to an HTTPError.")
            continue

        if cat.ngals > 0:
            cat = transient.associate(cat, cosmo, condition_host_props=condition_host_props_cat)

            if transient.best_host != -1:
                best_idx = transient.best_host
                second_best_idx = transient.second_best_host

                print_props = ['name', 'ra', 'dec', 'total_posterior']
                condition_props = list(priors.keys())

                log_host_properties(logger, transient.name, cat, best_idx, 
                                    Fore.BLUE+f"\nProperties of best host (in {cat_name} {cat_release})", print_props, calc_host_props,
                                    condition_props)
              
                # Set additional metadata
                result.update({
                    "host_ra": cat.galaxies['ra'][best_idx],
                    "host_dec": cat.galaxies['dec'][best_idx],
                    "host_redshift_mean": cat.galaxies['redshift_mean'][best_idx],
                    "host_redshift_std": cat.galaxies['redshift_std'][best_idx],
                    "host_prob": cat.galaxies['total_posterior'][best_idx],
                    "best_cat": cat_name,
                    "best_cat_release": cat_release,
                    "smallcone_posterior": transient.smallcone_posterior,
                    "missedcat_posterior": transient.missedcat_posterior,
                })
            
                logger.info(
                        f"Chosen galaxy has RA, DEC = {result['host_ra']:.6f}, {result['host_dec']:.6f}"
                    )

    if transient.best_host == -1:
        logger.info("No good host found!")

    return result




def run_prost(transient, output_dir, save=False):
    """
    Runs PROST to find the host galaxy of a transient.

    Parameters:
        transient (Transient): The transient object containing coordinates.
        output_dir (str): Directory to save PROST output.
        save (bool): Whether to save PROST data to output_dir

    Returns:
        list: A list of host galaxy associations from GHOST.
    """
    # Prost redshift, offset, and Absolute Mag Priors and Likelihoods

    if transient.transient_redshift == None:
        # No redshift prior
        priorfunc_offset = uniform(loc=0, scale=10) # go back to 10
        priorfunc_absmag = uniform(loc=-30, scale=20)
    
        likefunc_offset = gamma(a=0.75)
        likefunc_absmag = uniform(loc=-30, scale=20) #SnRateAbsmag(a=-25, b=20)
    
        priors = {"offset": priorfunc_offset, "absmag": priorfunc_absmag}
        likes = {"offset": likefunc_offset, "absmag": likefunc_absmag}

    else:
        priorfunc_z =  uniform(loc=0.0001, scale=0.6)
        priorfunc_offset = uniform(loc=0, scale=10) 
        priorfunc_absmag = uniform(loc=-30, scale=20)
    
        likefunc_offset = gamma(a=0.75)
        likefunc_absmag = uniform(loc=-30, scale=20) 
    
        priors = {"offset": priorfunc_offset, "absmag": priorfunc_absmag, "z": priorfunc_z}
        likes = {"offset": likefunc_offset, "absmag": likefunc_absmag}
    

    hosts = ProstSingleTransient(
        SN_obj=transient,
        glade_catalog=pd.read_csv(f'{PROST_ROOT}/GLADE+_HyperLedaSizes_mod_withz.csv.gz'),
        n_samples=1000,
        priors=priors,
        likes=likes,
        cosmo=cosmo,
        catalogs={'panstarrs': 'dr2',
                  'glade': 'latest',
                  'decals': 'dr10',               
                 } 
    )

    

    # Ensure output directory exists
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
    
        # Save the host associations to a CSV file
        now = datetime.now()
        date_str = f"{now.year}{now.month:02d}{now.day:02d}"
        output_csv = os.path.join(output_dir, f"hosts_{date_str}.csv")
    
        if os.path.exists(output_csv):
            with open(output_csv, "a", newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows(zip([hosts['name']], [hosts['host_ra']], [hosts['host_dec']], [hosts['host_redshift_mean']],
                                     [hosts['host_redshift_std']], [hosts['host_prob']], [hosts['smallcone_posterior']],
                                     [hosts['missedcat_posterior']], [hosts['best_cat']], [hosts['best_cat_release']]))
    
        

        else:
            # Create new file
            with open(output_csv, "w") as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(["Name","RA","Dec","z_mean","z_std","host_prob","smallcone_prob","missedcat_prob","best_cat", "best_cat_DR"])    
                writer.writerows(zip([hosts['name']], [hosts['host_ra']], [hosts['host_dec']], [hosts['host_redshift_mean']],
                                     [hosts['host_redshift_std']], [hosts['host_prob']], [hosts['smallcone_posterior']],
                                     [hosts['missedcat_posterior']], [hosts['best_cat']], [hosts['best_cat_release']]))
    

    return hosts
