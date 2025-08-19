import sys
import csv
import astropy.units as u  
import math
import os
import pickle
import time
import h5py
import numpy as np
from sedpy.observate import load_filters
import pandas
import glob
from classes import Transient, Filter, Host
from astropy.coordinates import SkyCoord
from mwebv_host import get_mwebv
import plot_best_models
import shutil
import time
from get_host_images import survey_list
from photutils.aperture import SkyEllipticalAperture
from sedpy.observate import load_filters

from prospect.utils.obsutils import fix_obs
from prospect.models import priors
from prospect.models.sedmodel import PolySpecModel
from prospect.sources import FastStepBasis
from prospect.fitting import fit_model as fit_model_prospect
from prospect.fitting import lnprobfn
from prospect.io import write_results as writer
from prospect.io.write_results import write_h5_header, write_obs_to_h5
import prospect.io.read_results as reader
from prospect.sources import FastStepBasis
from prospect.utils.plotting import quantile

import sedpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial
import matplotlib.ticker as mticker
import json
from astropy.cosmology import WMAP9 as cosmo
import prospect

SBIPP_ROOT = os.environ.get("SBIPP_ROOT")
SBIPP_PHOT_ROOT = os.environ.get("SBIPP_PHOT_ROOT")
SBIPP_TRAINING_ROOT = os.environ.get("SBIPP_TRAINING_ROOT")
SED_OUTPUT_ROOT = os.environ.get("SED_OUTPUT_ROOT")

def build_model_nonparam(obs=None, **extras):
    """prospector-alpha"""
    

    # -------------
    # MODEL_PARAMS
    model_params = {}

    # --- BASIC PARAMETERS ---
    model_params["zred"] = {
        "N": 1,
        "isfree": True,
        "init": 0.5,
        "prior": priors.FastUniform(a=0.0, b=1.5*1e-3) #priors.FastTruncatedNormal(a=0.0, b=0.6, mu=0.163, sig=0.052)
    } # ANYA: based on YSE DR1 mu and sigma

    fit_order = [
        "zred",
        "logmass",
        "logzsol",
        "logsfr_ratios",
        "dust2",
        "dust_index",
        "dust1_fraction",
        "log_fagn",
        "log_agn_tau",
        "duste_qpah",
        "duste_umin",
        "log_duste_gamma",
    ]

    model_params["logmass"] = {
        "N": 1,
        "isfree": True,
        "init": 8.0,
        "units": "Msun",
        "prior": priors.FastUniform(a=6.0, b=12.0),
    }

    model_params["logzsol"] = {
        "N": 1,
        "isfree": True,
        "init": -0.5,
        "units": r"$\log (Z/Z_\odot)$",
        "prior": priors.FastUniform(a=-1.98, b=0.19),
    }

    model_params["imf_type"] = {
        "N": 1,
        "isfree": False,
        "init": 1,  # 1 = chabrier
        "units": None,
        "prior": None,
    }
    model_params["add_igm_absorption"] = {"N": 1, "isfree": False, "init": True}
    model_params["add_agb_dust_model"] = {"N": 1, "isfree": False, "init": True}
    model_params["pmetals"] = {"N": 1, "isfree": False, "init": -99}

    # --- SFH ---
    nbins_sfh = 7
    model_params["sfh"] = {"N": 1, "isfree": False, "init": 3}
    model_params["logsfr_ratios"] = {
        "N": 6,
        "isfree": True,
        "init": 0.0,
        "prior": priors.FastTruncatedEvenStudentTFreeDeg2(
            hw=np.ones(6) * 5.0, sig=np.ones(6) * 0.3
        ),
    }

    # add redshift scaling to agebins, such that
    # t_max = t_univ
    def zred_to_agebins(zred=None, **extras):
        amin = 7.1295
        nbins_sfh = 7
        tuniv = cosmo.age(zred)[0].value * 1e9
        tbinmax = tuniv * 0.9
        if zred <= 3.0:
            agelims = (
                [0.0, 7.47712]
                + np.linspace(8.0, np.log10(tbinmax), nbins_sfh - 2).tolist()
                + [np.log10(tuniv)]
            )
        else:
            agelims = np.linspace(amin, np.log10(tbinmax), nbins_sfh).tolist() + [
                np.log10(tuniv)
            ]
            agelims[0] = 0

        agebins = np.array([agelims[:-1], agelims[1:]])
        return agebins.T

    def logsfr_ratios_to_masses(
        logmass=None, logsfr_ratios=None, agebins=None, **extras
    ):
        """This converts from an array of log_10(SFR_j / SFR_{j+1}) and a value of
        log10(\Sum_i M_i) to values of M_i.  j=0 is the most recent bin in lookback
        time.
        """
        nbins = agebins.shape[0]
        sratios = 10 ** np.clip(logsfr_ratios, -100, 100)
        dt = 10 ** agebins[:, 1] - 10 ** agebins[:, 0]
        coeffs = np.array(
            [
                (1.0 / np.prod(sratios[:i]))
                * (np.prod(dt[1 : i + 1]) / np.prod(dt[:i]))
                for i in range(nbins)
            ]
        )
        m1 = (10**logmass) / coeffs.sum()

        return m1 * coeffs

    model_params["mass"] = {
        "N": 7,
        "isfree": False,
        "init": 1e6,
        "units": r"M$_\odot$",
        "depends_on": logsfr_ratios_to_masses,
    }

    model_params["agebins"] = {
        "N": 7,
        "isfree": False,
        "init": zred_to_agebins(np.atleast_1d(0.5)),
        "prior": None,
        "depends_on": zred_to_agebins,
    }

    # --- Dust Absorption ---
    model_params["dust_type"] = {
        "N": 1,
        "isfree": False,
        "init": 4,
        "units": "FSPS index",
    }
    model_params["dust1_fraction"] = {
        "N": 1,
        "isfree": True,
        "init": 1.0,
        "prior": priors.FastTruncatedNormal(a=0.0, b=2.0, mu=1.0, sig=0.3),
    }

    model_params["dust2"] = {
        "N": 1,
        "isfree": True,
        "init": 0.0,
        "units": "",
        "prior": priors.FastTruncatedNormal(a=0.0, b=4.0, mu=0.3, sig=1.0),
    }

    def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
        return dust1_fraction * dust2

    model_params["dust1"] = {
        "N": 1,
        "isfree": False,
        "depends_on": to_dust1,
        "init": 0.0,
        "units": "optical depth towards young stars",
        "prior": None,
    }
    model_params["dust_index"] = {
        "N": 1,
        "isfree": True,
        "init": 0.7,
        "units": "",
        "prior": priors.FastUniform(a=-1.0, b=0.2),
    }

    # --- Nebular Emission ---
    model_params['nebemlineinspec'] = {"N": 1, "isfree": False, "init": False}
    #model_params["add_neb_emission"] = {"N": 1, "isfree": False, "init": True}
    #model_params["add_neb_continuum"] = {"N": 1, "isfree": False, "init": True}
    model_params["gas_logz"] = {
        "N": 1,
        "isfree": False,
        "init": 0.0,
        "units": r"log Z/Z_\odot",
        "prior": priors.FastUniform(a=-2.0, b=0.5),
    }
    model_params["gas_logu"] = {
        "N": 1,
        "isfree": False,
        "init": -1.0,
        "units": r"Q_H/N_H",
        "prior": priors.FastUniform(a=-4, b=-1),
    }

    # --- AGN dust ---
    model_params["add_agn_dust"] = {"N": 1, "isfree": False, "init": True}

    model_params["log_fagn"] = {
        "N": 1,
        "isfree": True,
        "init": -7.0e-5,
        "prior": priors.FastUniform(a=-5.0, b=-4.9),
    }

    def to_fagn(log_fagn=None, **extras):
        return 10**log_fagn

    model_params["fagn"] = {"N": 1, "isfree": False, "init": 0, "depends_on": to_fagn}

    model_params["log_agn_tau"] = {
        "N": 1,
        "isfree": True,
        "init": np.log10(20.0),
        "prior": priors.FastUniform(a=np.log10(15.0), b=np.log10(15.1)),
    }

    def to_agn_tau(log_agn_tau=None, **extras):
        return 10**log_agn_tau

    model_params["agn_tau"] = {
        "N": 1,
        "isfree": False,
        "init": 0,
        "depends_on": to_agn_tau,
    }

    # --- Dust Emission ---
    model_params["duste_qpah"] = {
        "N": 1,
        "isfree": True,
        "init": 2.0,
        "prior": priors.FastTruncatedNormal(a=0.9, b=1.1, mu=2.0, sig=2.0),
    }

    model_params["duste_umin"] = {
        "N": 1,
        "isfree": True,
        "init": 1.0,
        "prior": priors.FastTruncatedNormal(a=0.9, b=1.1, mu=1.0, sig=10.0),
    }

    model_params["log_duste_gamma"] = {
        "N": 1,
        "isfree": True,
        "init": -2.0,
        "prior": priors.FastTruncatedNormal(a=-2.1, b=-1.9, mu=-2.0, sig=1.0),
    }

    def to_duste_gamma(log_duste_gamma=None, **extras):
        return 10**log_duste_gamma

    model_params["duste_gamma"] = {
        "N": 1,
        "isfree": False,
        "init": 0,
        "depends_on": to_duste_gamma,
    }

    # ---- Units ----
    model_params["peraa"] = {"N": 1, "isfree": False, "init": False}

    model_params["mass_units"] = {"N": 1, "isfree": False, "init": "mformed"}

    tparams = {}
    for i in fit_order:
        tparams[i] = model_params[i]
    for i in list(model_params.keys()):
        if i not in fit_order:
            tparams[i] = model_params[i]
    model_params = tparams

    return PolySpecModel(model_params)


def get_best_fit_SED(resultpars, obs):
    
    maggies = obs['maggies']
    maggies_unc = obs['maggies_unc']
    obs_filters = obs['filters']
    phot_obs_wave = obs['phot_wave']
    zred = obs['redshift']
        
    model = build_model_nonparam(obs)
    stellarpop = FastStepBasis(zcontinuous=2, compute_vega_mags=False)
    
    
    for i in range(2500):
        theta = resultpars["chain"][np.random.choice(np.arange(np.shape(resultpars["chain"])[0])), :]
        if obs["redshift"] != None:
            theta[0] = obs["redshift"]            
    
        if i == 0:
            best_spec, best_phot, mfrac = model.predict(theta, obs=obs, sps=stellarpop)
    
            best_phot_store = best_phot[:]
            best_spec_store = best_spec[:]
            mfrac_store = []
            theta_zred = []
        else:
            best_spec_single, best_phot_single, mfrac_single = model.predict(theta, obs=obs, sps=stellarpop)
    
            # iteratively update the mean
            best_spec = (best_spec * i + best_spec_single) / (i + 1)
            best_phot = (best_phot * i + best_phot_single) / (i + 1)
            mfrac = (mfrac * i + mfrac_single) / (i + 1)
            mfrac_store.append(mfrac)
            theta_zred.append(theta[0])
            best_phot_store = np.vstack([best_phot_store, best_phot_single])
            best_spec_store = np.vstack([best_spec_store, best_spec_single])
    
    best_phot = np.median(best_phot_store, axis=0)
    best_spec = np.median(best_spec_store, axis=0)
    best_mfrac = np.median(mfrac_store)
    mod_zred = np.median(theta_zred)
    phot_16, phot_84 = np.percentile(best_phot_store, [16, 84], axis=0)
    spec_16, spec_84 = np.percentile(best_spec_store, [16, 84], axis=0)
    
    
    filternames = [filter.name for filter in obs["filters"]]
    obs["filters"] = sedpy.observate.load_filters(filternames)
    rest_wavelength = stellarpop.wavelengths.copy()



    model_SED_dict = {
        'best_spec': best_spec,
        'spec_16': spec_16,
        'spec_84': spec_84,
        'best_phot': best_phot,
        'phot_16': phot_16,
        'phot_84': phot_84,
        'obs_phot': maggies,
        'obs_phot_unc': maggies_unc,
        'obs_filters': obs_filters,
        'phot_wave': phot_obs_wave,
        'obs_redshift': zred,
        'rest_spec_wavelengths': rest_wavelength,
        'mfrac': best_mfrac,
        'mod_zred': mod_zred
    }

    print('Found best fit model SED.')

    return model_SED_dict


def getPercentiles(chain, parnames, quantity="zred", percents=[15.9, 50.0, 84.1]):
    """get the 16/50/84th percentile for a scalar output
    that does not need transform functions
    (e.g., mass, dust, etc).
    """
    try:
        npix = chain[:, parnames.index(quantity)].shape[0]
    except ValueError:
        print('"' + quantity + '" does not exist in the output.')
        return
    p = np.percentile(chain[:, parnames.index(quantity)], q=percents)
    return p


def resample_chain(resultpars, model_SED_dict, ncalc=2500):
    chain = resultpars["chain"]
    parnames = resultpars['theta_labels']
    
    npoints = chain.shape[0]
    sample_idx = np.random.choice(np.arange(npoints),size=ncalc,replace=True)
    thetas = chain[sample_idx,:]

    logzsol = thetas[:,parnames.index('logzsol')]
    zred = thetas[:,parnames.index('zred')]
    log_fagn = thetas[:,parnames.index('log_fagn')]
    log_agn_tau = thetas[:,parnames.index('log_agn_tau')]

    return logzsol, zred, log_fagn, log_agn_tau


def stellar_mass_calc(resultpars, model_SED_dict, ncalc=2500):
    chain = resultpars["chain"]
    parnames = resultpars['theta_labels']
    mfrac = model_SED_dict["mfrac"]
    
    npoints = chain.shape[0]
    sample_idx = np.random.choice(np.arange(npoints),size=ncalc,replace=True)
    thetas = chain[sample_idx,:]

    surviving_mass = []
    for theta in thetas:
        surviving_mass.append(theta[parnames.index('logmass')] + np.log10(mfrac))

    return np.array(surviving_mass)


def dust_AV(resultpars, model_SED_dict, ncalc=2500):
    chain = resultpars["chain"]
    parnames = resultpars['theta_labels']
    
    npoints = chain.shape[0]
    sample_idx = np.random.choice(np.arange(npoints),size=ncalc,replace=True)
    thetas = chain[sample_idx,:]

    dust1_frac = thetas[:,parnames.index('dust1_fraction')]
    dust2 = thetas[:,parnames.index('dust2')]

    dust = (dust1_frac*dust2 + dust2)*1.086
    
    return dust



def zred_to_agebins(zred=None, nbins_sfh=7):
    tuniv = cosmo.age(zred).value[0]*1e9
    tbinmax = (tuniv*0.9)
    agelims = [0.0,7.4772] + np.linspace(8.0,np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv)]
    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T


def get_mwa(agebins, sfr):
    ages = 10**agebins
    dt = np.abs(ages[:, 1] - ages[:, 0])
    return np.sum(np.mean(ages, axis=1) * sfr * dt) / np.sum(sfr * dt) / 1e9  # in Gyr



def get_MWA_and_SFH(resultpars, model_SED_dict, ncalc=2500):
    """
    Currently only works when nbins_SFH=7!
    """
    chain = resultpars["chain"]
    parnames = resultpars['theta_labels']
    mfrac = model_SED_dict["mfrac"]
    
    npoints = chain.shape[0]
    sample_idx = np.random.choice(np.arange(npoints),size=ncalc,replace=True)
    thetas = chain[sample_idx,:]

    if model_SED_dict["obs_redshift"] is not None:
        zred = model_SED_dict["obs_redshift"]
    else:
        zred = model_SED_dict['mod_zred']

    nbins_sfh=7
    agebins = zred_to_agebins(zred=[zred],nbins_sfh=7)

    SFR = []
    sfr = np.zeros((ncalc, nbins_sfh)) # change to number of bins
            
    for i in range(ncalc): 
        sfr[i] = prospect.models.transforms.logsfr_ratios_to_sfrs(logmass=thetas[:, parnames.index('logmass')][i],
                                                                  logsfr_ratios=[thetas[:, parnames.index('logsfr_ratios_1')][i],
                                                                                 thetas[:, parnames.index('logsfr_ratios_2')][i],
                                                                                 thetas[:, parnames.index('logsfr_ratios_3')][i],
                                                                                 thetas[:, parnames.index('logsfr_ratios_4')][i],
                                                                                 thetas[:, parnames.index('logsfr_ratios_5')][i],
                                                                                 thetas[:, parnames.index('logsfr_ratios_6')][i]], 
                                                                  agebins=agebins)
        
        SFR.append(sfr[i])

    # Now we have a numpy array of the mass and SFR in each bin
    star_formation = [list(j) for j in SFR]
    npSFR = np.array(star_formation, dtype=float)
    MWA = []

    for p in npSFR:
        MWA.append(get_mwa(agebins, p))
    
    MWA_arr = np.array(MWA)

    # Calculate 100 Myr SFR
    SFR_100_Myr= np.median([npSFR[:,0],npSFR[:,1]], axis=0)


    return agebins, MWA_arr, npSFR, SFR_100_Myr



def star_formation_history(transient_name, agebins, npSFR, show_plot=False):
    nbins_sfh = 7
    t_lookback_log = [age[0] for age in agebins]
    t_lookback_log  = np.append(t_lookback_log, agebins[-1][-1]) # double last bin for plot purposes
    t_lookback_log[0]=0  # set age range for aesthetic purposes

    t_lookback  = (10**t_lookback_log)/1e9 # de-log agebins (in yr) and convert to Gyr


    #Calculate the quantiles
    Q16_SFR = [quantile(npSFR[:, i], percents=16, weights = None)
                 for i in range(nbins_sfh)]
             
    Q50_SFR = [quantile(npSFR[:, i], percents=50, weights = None)
                 for i in range(nbins_sfh)]

    Q84_SFR = [quantile(npSFR[:, i], percents=84, weights=None)
                 for i in range(nbins_sfh)]      


    # 16th percentile of SFR
    SFH_lower = np.insert(Q16_SFR, 0, Q16_SFR[0], axis=0)
    # 50th percentile of SFR
    SFH = np.insert(Q50_SFR, 0, Q50_SFR[0], axis=0)
    # 84th percentile of SFR
    SFH_upper = np.insert(Q84_SFR, 0, Q84_SFR[0], axis=0)


    if show_plot:

        fig = plt.figure(figsize=[10,10])

        fig, ax = plt.subplots(figsize=[8,6])
        plt.title(f'{transient_name} Star Formation History', fontsize = 20)
        plt.xlabel('t$_{lookback}$ (Gyr)', fontsize = 20)
        plt.ylabel('SFR ($M_{\odot}$yr$^{-1}$)', fontsize = 20)
        plt.rc('xtick', labelsize=15) 
        plt.rc('ytick', labelsize=15)
        mpl.rcParams['xtick.major.size'] = 5
        mpl.rcParams['xtick.major.width'] = 3
        mpl.rcParams['xtick.minor.width'] = 3
        mpl.rcParams['ytick.major.size'] = 6
        mpl.rcParams['ytick.minor.width'] = 2
        mpl.rcParams['ytick.major.width'] = 3
        ax.tick_params(axis='x', which='minor')
        
        # use step function to generate the SFH plot
        plt.step(t_lookback, SFH_lower, alpha = 1, c = 'lightgreen', linewidth=3)
        plt.step(t_lookback, SFH_upper, alpha = 1, c = 'lightgreen', linewidth=3)
        plt.step(t_lookback, SFH, alpha = 1.0, c = 'green', linewidth=3)
        
        # Since it's the lookback time, we want the x-axis limits to be flipped
        plt.xlim(t_lookback[-1], 0.01) # setting max for aesthetic purposes
        #plt.xscale('log')
        plt.yscale('log')
        
        
        
        for ax in fig.get_axes():
            ax.tick_params(axis='both', labelsize=20, length=12, width=2, direction='in')
            ax.set_xscale('log')
            ax.set_xticks([0.01, 0.1, 1, 10])
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.get_major_formatter().set_scientific(False)
            ax.xaxis.get_major_formatter().set_useOffset(False)
            #ax.xaxis.set_minor_formatter(mticker.ScalarFormatter(False))
            
            
            ax.yaxis.labelpad = -10
        
        # shading the regions between 16th and 84th percentiles
        plt.fill_between(t_lookback,SFH_lower,SFH_upper, where=None,interpolate=False,step='pre',color="lightgreen")

        plt.savefig(f'./data/sed_output/SFH_plots/{transient_name}_SFH.png', dpi=300)
        plt.show()


    return t_lookback, SFH, SFH_lower, SFH_upper



def save_all(
    transient,
    prospector_output,
    model_components,
    observations,
    sed_output_root=SED_OUTPUT_ROOT,
    aperture_type = None, **kwargs):
    
    # write the results
    hdf5_file_name = (
        f"{sed_output_root}/{transient.name}/{transient.name}_{aperture_type}.h5"
    )
    hdf5_file = hdf5_file_name
   
    if not os.path.exists(f"{sed_output_root}/{transient.name}"):
        os.makedirs(f"{sed_output_root}/{transient.name}/")
    if os.path.exists(hdf5_file):
        # prospector won't overwrite, which causes problems
        os.remove(hdf5_file)

    hf = h5py.File(hdf5_file_name, "w")
    print(hdf5_file_name)

    sdat = hf.create_group("sampling")
    sdat.create_dataset("chain", data=prospector_output["sampling"][0]["samples"])
    sdat.attrs["theta_labels"] = json.dumps(
        list(model_components["model"].theta_labels())
    )

    # High level parameter and version info
    write_h5_header(hf, {}, model_components["model"])

    # ----------------------
    # Observational data
    del observations['filternames']
    write_obs_to_h5(hf, observations)
    hf.flush()
    print('Saved to h5 file')
 
    resultpars, obs, _ = reader.results_from(hdf5_file, dangerous=False)

    # Get Best-fit Model SED, mfrac, observed or modeled redshift
    host_output_dict = get_best_fit_SED(resultpars, obs)

    # Calculate stellar mass
    stellar_mass_chain = stellar_mass_calc(resultpars, host_output_dict, ncalc=2500)

    # Calculate total dust
    dust_AV_chain = dust_AV(resultpars, host_output_dict, ncalc=2500)

    agebins, MWA_chain, npSFR_chain, SFR_100_Myr_chain = get_MWA_and_SFH(resultpars, host_output_dict, ncalc=2500)

    # Get SFH
    t_lookback, SFH, SFH_lower, SFH_upper = star_formation_history(transient.name, agebins, npSFR_chain, show_plot=False)

    # Get raw properties that we want
    logzsol_chain, zred_chain, log_fagn_chain, log_agn_tau_chain = resample_chain(resultpars, host_output_dict, ncalc=2500)


    # Save chains
    host_output_dict['logzsol_chain'] = logzsol_chain
    host_output_dict['zred_chain'] = zred_chain
    host_output_dict['log_fagn_chain'] = log_fagn_chain
    host_output_dict['log_agn_tau_chain'] = log_agn_tau_chain
    host_output_dict['dust_AV_chain'] = dust_AV_chain
    host_output_dict['stellar_mass_chain'] = stellar_mass_chain
    host_output_dict['MWA_chain'] = MWA_chain
    host_output_dict['SFR_100_Myr_chain'] = SFR_100_Myr_chain



    # Calculate Percentiles
    host_output_dict['logzsol_perc'] = np.percentile(logzsol_chain, q=[15.9, 50.0, 84.1])
    host_output_dict['zred_perc'] = np.percentile(zred_chain, q=[15.9, 50.0, 84.1])
    host_output_dict['log_fagn_perc'] = np.percentile(log_fagn_chain, q=[15.9, 50.0, 84.1])
    host_output_dict['log_agn_tau_perc'] = np.percentile(log_agn_tau_chain, q=[15.9, 50.0, 84.1])
    host_output_dict['dust_AV_perc'] = np.percentile(dust_AV_chain, q=[15.9, 50.0, 84.1])
    host_output_dict['stellar_mass_perc'] = np.percentile(stellar_mass_chain, q=[15.9, 50.0, 84.1])
    host_output_dict['MWA_perc'] = np.percentile(MWA_chain, q=[15.9, 50.0, 84.1])
    host_output_dict['SFR_100_Myr_perc'] = np.percentile(SFR_100_Myr_chain, q=[15.9, 50.0, 84.1])


    # Save SFH
    host_output_dict['t_lookback_for_SFH'] = t_lookback
    host_output_dict['SFH_median'] = SFH
    host_output_dict['SFH_lower'] = SFH_lower
    host_output_dict['SFH_upper'] = SFH_upper


    # Save 
    # write the results
    npy_file_name = (f"{sed_output_root}/{transient.name}/{transient.name}_{aperture_type}.npy")

    np.save(npy_file_name, host_output_dict)
    print(f'Host model and stellar population properties saved to {npy_file_name}.')

    return host_output_dict
    
    

    