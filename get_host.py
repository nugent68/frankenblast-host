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

from classes import Transient as BlastTransient
from classes import Filter, Host
from mwebv_host import get_mwebv
from get_host_images import download_and_save_cutouts, survey_list, get_cutouts
from create_apertures import construct_aperture
from do_photometry import do_global_photometry

import csv
import shutil
import glob

from astro_prost.diagnose import plot_match
from astro_prost.helpers import GalaxyCatalog, Transient
from astro_prost.helpers import PriorzObservedTransients, SnRateAbsmag

import pathlib
from mpire import WorkerPool
import time
from urllib.error import HTTPError
import astropy.units as u
import numpy as np
import requests
import importlib.resources as pkg_resources
import importlib



def ProstSingleTransient(
    SN_obj,
    n_samples,
    verbose,
    priors,
    likes,
    cosmo,
    catalogs,
):
    """Associates a transient with its most likely host galaxy.

    Parameters
    ----------
    blast_obj: FrankenBlast transient class instantiation
    n_samples : int
        Number of samples for the monte-carlo sampling of associations.
    verbose : int
        Level of logging during run (can be 0, 1, or 2).

    priors: Dictionary of priors
        z : scipy stats continuous distribution
            Prior distribution on redshift. This class can be user-defined
            but needs .sample(size=n) and .pdf(x) functions.
        offset : scipy stats continuous distribution
            Prior distribution on fractional offset.
        absmag : scipy stats continuous distribution
            Prior distribution on host absolute magnitude.
    likes: 
        offset : scipy stats continuous distribution
            Likelihood distribution on fractional offset.
        absmag : scipy stats continuous distribution.
            Likelihood distribution on host absolute magnitude.
    cosmo : astropy cosmology
        Assumed cosmology for the run (defaults to LambdaCDM if unspecified).
    catalogs : list
        List of source catalogs to query (can include 'glade', 'decals', or 'panstarrs').
    cat_cols : boolean
        If true, concatenates the source catalog fields to the returned dataframe.
    Returns
    -------
    tuple
        Properties of the best host galaxy
    """
    try:
        transient = Transient(
            name=SN_obj.name,
            position=SN_obj.coordinates,
            redshift=SN_obj.transient_redshift,
            n_samples=n_samples,
        )
        no_redshift_fit = False
    except (KeyError, TypeError, AttributeError):
        transient = Transient(
            name=SN_obj.name,
            position=SN_obj.coordinates,
            n_samples=n_samples,
        )
        no_redshift_fit = True

    if verbose > 0:
        print(
            f"Associating {transient.name} at RA, DEC = "
            f"{transient.position.ra.deg:.6f}, {transient.position.dec.deg:.6f}"
        )

    
    transient.set_prior("redshift", priors["z"])
    transient.set_prior("offset", priors["offset"])
    transient.set_prior("absmag", priors["absmag"])

    transient.set_likelihood("offset", likes["offset"])
    transient.set_likelihood("absmag", likes["absmag"])

    transient.gen_z_samples(n_samples=n_samples)

    (
        SN_name, best_prob, best_ra, best_dec, best_zred, best_zred_std,best_cat,
        smallcone_prob, missedcat_prob, bayes_factor
    ) = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )

    best_cat = ""

    best_host_all_cat = []

    for cat_name in catalogs:
        glade_catalog = pd.read_csv('./data/prost_data/GLADE+_HyperLedaSizes_mod_withz.csv')
        
        cat = GalaxyCatalog(name=cat_name, n_samples=n_samples, data=glade_catalog)

        try:
            cat.get_candidates(transient, time_query=True, verbose=verbose, cosmo=cosmo)
        except requests.exceptions.HTTPError:
            print(f"Candidate retrieval failed for {transient.name} in catalog {cat_name}.")
            continue

        if cat.ngals > 0:
            cat = transient.associate(cat, cosmo, no_redshift_fit=no_redshift_fit, verbose=verbose)

            if transient.best_host != -1:
                best_idx = transient.best_host
                second_best_idx = transient.second_best_host

                if verbose >= 2:
                    print_cols = [
                        "objID",
                        "z_prob",
                        "offset_prob",
                        "absmag_prob",
                        "total_prob",
                        "ra",
                        "dec",
                        "offset_arcsec",
                        "z_best_mean",
                        "z_best_std",
                    ]
                    print("Properties of best host:")
                    for key in print_cols:
                        print(key)
                        print(cat.galaxies[key][best_idx])

                    print("Properties of second best host:")
                    for key in print_cols:
                        print(key)
                        print(cat.galaxies[key][second_best_idx])

                best_objid = np.int64(cat.galaxies["objID"][best_idx])
                best_prob = cat.galaxies["total_prob"][best_idx]
                best_ra = cat.galaxies["ra"][best_idx]
                best_dec = cat.galaxies["dec"][best_idx]
                best_zred = cat.galaxies["z_best_mean"][best_idx]
                best_zred_std = cat.galaxies["z_best_std"][best_idx]

                second_best_objid = np.int64(cat.galaxies["objID"][second_best_idx])
                second_best_prob = cat.galaxies["total_prob"][second_best_idx]
                #second_best_ra = cat.galaxies["ra"][second_best_idx]
                #second_best_dec = cat.galaxies["dec"][second_best_idx]

                best_cat = cat_name
                query_time = cat.query_time
                smallcone_prob = transient.smallcone_prob
                missedcat_prob = transient.missedcat_prob
                bayes_factor= best_prob/second_best_prob
                
                best_host_all_cat.append([SN_obj.name,best_prob,best_ra,best_dec,best_zred,best_zred_std,best_cat,
                                          smallcone_prob,missedcat_prob,bayes_factor])

                print(best_host_all_cat)

                if verbose > 0:
                    print(
                        f"Chosen {cat_name} galaxy has catalog ID of {best_objid}"
                        f" and RA, DEC = {best_ra:.6f}, {best_dec:.6f}"
                    )
                    #print(SN_obj.name,best_prob,best_ra,best_dec,best_zred,best_zred_std,best_cat,
                    #       smallcone_prob,missedcat_prob,bayes_factor)
                if verbose > 1:
                    try:
                        plot_match(
                            [best_ra],
                            [best_dec],
                            cat.galaxies["z_best_mean"][best_idx],
                            cat.galaxies["z_best_std"][best_idx],
                            transient.position.ra.deg,
                            transient.position.dec.deg,
                            transient.name,
                            transient.redshift,
                            0,
                            f"{transient.name}_{cat_name}",
                        )
                    except HTTPError:
                        print("Couldn't get an image. Waiting 60s before moving on.")
                        time.sleep(60)
                        continue

    if (transient.best_host == -1) and (verbose > 0):
        print('No host found!')

        

    best_host_all_cat = np.array(best_host_all_cat)

    # We return the most likely host from all the catalogs queried
    best = best_host_all_cat[np.where(np.float64(best_host_all_cat[:,1]) == np.max(np.float64(best_host_all_cat[:,1])))][0]
    
    return (
        best[0],
        np.float64(best[1]),
        np.float64(best[2]),
        np.float64(best[3]),
        np.float64(best[4]),
        np.float64(best[5]),
        best[6],
        np.float64(best[7]),
        np.float64(best[8]),
        np.float64(best[9]),
            )



def run_prost(transient, output_dir):
    """
    Runs PROST to find the host galaxy of a transient.

    Parameters:
        transient (Transient): The transient object containing coordinates.
        output_dir (str): Directory to save GHOST output.

    Returns:
        list: A list of host galaxy associations from GHOST.
    """
    # Prost redshift, offset, and Absolute Mag Priors and Likelihoods
    priorfunc_z =  uniform(loc=0.0001, scale=0.6)
    priorfunc_offset = uniform(loc=0, scale=10) # go back to 10
    priorfunc_absmag = uniform(loc=-30, scale=20)

    likefunc_offset = gamma(a=0.75)
    likefunc_absmag = uniform(loc=-30, scale=20) #SnRateAbsmag(a=-25, b=20)

    priors = {"offset": priorfunc_offset, "absmag": priorfunc_absmag, "z": priorfunc_z}
    likes = {"offset": likefunc_offset, "absmag": likefunc_absmag}
    

    # Run Prost to get host associations
    hosts = ProstSingleTransient(
        SN_obj=transient,
        n_samples=1000,
        verbose=1,
        priors=priors,
        likes=likes,
        cosmo=cosmo,
        catalogs=["decals", "panstarrs"],
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the host associations to a CSV file
    #now = datetime.now()
    #date_str = f"{now.year}{now.month:02d}{now.day:02d}"
    #output_csv = os.path.join(output_dir, f"hosts_{date_str}.csv")

    output_csv = os.path.join(output_dir, "SLSN_most_host_hosts.csv")
    
    if os.path.exists(output_csv):
        with open(output_csv, "a", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(zip([hosts[0]], [hosts[2]], [hosts[3]], [hosts[4]], 
                                 [hosts[5]], [hosts[1]], [hosts[7]], [hosts[8]],[hosts[9]],[hosts[6]]))
        

    else:
        # Create new file
        with open(output_csv, "w") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(["Name","RA","Dec","z_mean","z_std","host_prob","smallcone_prob","missedcat_prob","bayes","best_cat"])    
            writer.writerows(zip([hosts[0]], [hosts[2]], [hosts[3]], [hosts[4]], 
                                 [hosts[5]], [hosts[1]], [hosts[7]], [hosts[8]],[hosts[9]],[hosts[6]]))
    

    return hosts
