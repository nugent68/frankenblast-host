import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from scipy.interpolate import interp1d
import os
import math
import pickle
import h5py
import json
import copy
import time
from sedpy.observate import load_filters
import sbi_pp_photoz

from prospect.utils.obsutils import fix_obs
from prospect.models import priors
from prospect.models.sedmodel import PolySpecModel
from prospect.sources import FastStepBasis
from prospect.fitting import fit_model as fit_model_prospect
from prospect.fitting import lnprobfn
from prospect.io import write_results as writer
from prospect.io.write_results import write_h5_header, write_obs_to_h5
import prospect.io.read_results as reader
from sedpy.observate import Filter
from postprocess_sbi import save_all as psbi_save

import extinction
import sbi_pp
# ANYA - fix the below later for Prospector output
# import postprocess_prosp as pp

SBIPP_ROOT = os.environ.get("SBIPP_ROOT")
SBIPP_PHOT_ROOT = os.environ.get("SBIPP_PHOT_ROOT")
SBIPP_TRAINING_ROOT = os.environ.get("SBIPP_TRAINING_ROOT")
SED_OUTPUT_ROOT = os.environ.get("SED_OUTPUT_ROOT")


run_params = {
    "nmc":50,  # 50number of MC samples
    "nposterior": 50,  # 50 number of posterior samples per MC drawn
    "np_baseline": 2500,  # 500 number of posterior samples used in baseline SBI
    "ini_chi2": 5,  # chi^2 cut usedi in the nearest neighbor search
    "max_chi2": 5000,  # the maximum chi^2 to reach in case we Incremently increase the chi^2
    # in the case of insufficient neighbors
    "noisy_sig": 10,  # 3 deviation from the noise model, above which the measuremnt is taked as OOD
    "tmax_per_obj": 120000,  # 120000max time spent on one object / mc sample in secs
    "tmax_all": 600000,  # 60 max time spent on all mc samples in mins
    "outdir": "output",  # output directory
    "verbose": True,
    "tmax_per_iter": 60,#60
}


def maggies_to_asinh(x):
    """asinh magnitudes"""
    a = 2.50 * np.log10(np.e)
    mu = 35.0
    return -a * math.asinh((x / 2.0) * np.exp(mu / a)) + mu


def mJy_to_maggies(flux_mJy):
    """
    Converts spectral flux density from mJy to units of maggies.
    """
    return flux_mJy * 10 ** (-0.4 * 23.9)


def run_training_set(fname, verbose=True):
    print("""Loading training sets from data files...""")
    with open(
        os.path.join(SBIPP_TRAINING_ROOT, f"hatp_x_y_{fname}_global.pkl"), "rb"
    ) as handle:
        hatp_x_y_global = pickle.load(handle)
    with open(
        os.path.join(SBIPP_TRAINING_ROOT, f"y_train_{fname}_global.pkl"), "rb"
    ) as handle:
        y_train_global = pickle.load(handle)
    with open(
        os.path.join(SBIPP_TRAINING_ROOT, f"x_train_{fname}_global.pkl"), "rb"
    ) as handle:
        x_train_global = pickle.load(handle)
    if verbose:
        print("""Training sets loaded.""") 

    print(len(y_train_global))

    return hatp_x_y_global, y_train_global, x_train_global



def fit_sbi_pp(observations, all_filters=None, sbi_params=None, fname=None, n_filt_cuts=True):
    np.random.seed(100)  # make results reproducible

    uv_filters = ["GALEX_NUV", "GALEX_FUV", "SDSS_u", "DES_u"]
    opt_filters = [
        "SDSS_g",
        "SDSS_r",
        "SDSS_i",
        "SDSS_z",
        "PanSTARRS_g",
        "PanSTARRS_r",
        "PanSTARRS_i",
        "PanSTARRS_z",
        "PanSTARRS_y",
        "DES_g",
        "DES_r",
    ]
    ir_filters = [
        "WISE_W1",
        "WISE_W2",
        "WISE_W3",
        "WISE_W4",
        "2MASS_J",
        "2MASS_H",
        "2MASS_K",
    ]

    # toy noise model
    meds_sigs, stds_sigs = [], []

    for f in all_filters:
        toy_noise_x, toy_noise_y = np.loadtxt(
            f"data/SBI/snrfiles/{f.name}_magvsnr.txt", dtype=float, unpack=True
        )
        meds_sigs += [
            interp1d(
                toy_noise_x,
                1.0857 * 1 / toy_noise_y,
                kind="slinear",
                fill_value="extrapolate",  # (0.01,1.0),
                #bounds_error=False,
            )
        ]
        stds_sigs += [
            interp1d(
                toy_noise_x,
                1.0857 * 1 / toy_noise_y,
                kind="slinear",
                fill_value="extrapolate",  # (0.01,1.0),
                #bounds_error=False,
            )
        ]
    sbi_params["toynoise_meds_sigs"] = meds_sigs
    sbi_params["toynoise_stds_sigs"] = stds_sigs

    # a testing object of which the noises are OOD
    mags, mags_unc, filternames, wavelengths = (
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )

    has_uv, has_opt, has_ir = False, False, False
    for f in all_filters:
        if f.name in observations["filternames"]:
            iflt = np.array(observations["filternames"]) == f.name
            mags = np.append(mags, maggies_to_asinh(observations["maggies"][iflt]))
            mags_unc = np.append(
                mags_unc,
                2.5
                / np.log(10)
                * observations["maggies_unc"][iflt]
                / observations["maggies"][iflt],
            )
            if f.name in uv_filters:
                has_uv = True
            elif f.name in opt_filters:
                has_opt = True
            elif f.name in ir_filters:
                has_ir = True
        else:
            mags = np.append(mags, np.nan)
            mags_unc = np.append(mags_unc, np.nan) 
        filternames = np.append(filternames, f.name)
        wavelengths = np.append(wavelengths, f.wavelength_eff_angstrom)

    obs = {}
    obs[
        "mags"
    ] = mags  ##np.array([maggies_to_asinh(p) for p in observations['maggies']])
    obs[
        "mags_unc"
    ] = mags_unc  ##2.5/np.log(10)*observations['maggies_unc']/observations['maggies']
    obs["redshift"] = observations["redshift"]
    obs["wavelengths"] = wavelengths
    obs["filternames"] = filternames



    if n_filt_cuts and not has_opt and (not has_ir or not has_uv):
        print("not enough filters for reliable/fast inference")
        return {}, 1

    # prepare to pass the reconstructed model to sbi_pp
    hatp_x_y_global, y_train_global, x_train_global = run_training_set(fname, verbose=True)
    
    sbi_params["y_train"] = y_train_global
    sbi_params["theta_train"] = x_train_global
    sbi_params["hatp_x_y"] = hatp_x_y_global
    
    if obs['redshift'] != None:
        # Run SBI++ with the redshift
        print('Host has spectroscopic redshift.')
        chain, obs, flags = sbi_pp.sbi_pp(
            obs=obs, run_params=run_params, sbi_params=sbi_params
        )

    else:
        # Run SBI++ with to determine the redshift + other properties
        print('Host has no spectroscopic redshift.')
        chain, obs, flags = sbi_pp_photoz.sbi_pp(
            obs=obs, run_params=run_params, sbi_params=sbi_params
        )
    # pathological format as we're missing some stuff that prospector usually spits out
    output = {"sampling": [{"samples": chain[:, :], "eff": 100}, 0]}
    return output, 0



def fit_host(transient, sbi_params=None, fname=None, all_filters = None, mode='test', sbipp=True, 
            aperture_type='global', aperture=None, save=True):
    
    observations = build_obs(transient)
    model_components = build_model(observations)


    print('Loaded observations and model')
    print("starting model fit")

    if sbipp:
        posterior, errflag = fit_sbi_pp(observations, all_filters, sbi_params, fname, n_filt_cuts=True)
        prosp_output_dict = psbi_save(transient,posterior,model_components,observations,
                                      sed_output_root=SED_OUTPUT_ROOT,aperture_type = aperture_type)
    else:
        return("You are trying to run normal Prospector. This is not available yet")
        

    if errflag:
        return("not enough filters")

    return 'done'



def build_obs(transient, aperture_type='global', use_mag_offset=False):
    """
    This function is required by prospector and should return
    a dictionary defined by
    https://prospect.readthedocs.io/en/latest/dataformat.html.
    """

    photometry = transient.host_photometry

    if photometry is None:
        raise ValueError("No host photometry")

    if transient.host is None:
        raise ValueError("No host galaxy match")

    # ANYA: THIS WILL BREAK IF THERE IS JUST A PHOTOMETRIC REDSHIFT!
    z = transient.host.redshift

    # Apply redshift offset if necessary
    if z is not None and z < 0.015 and use_mag_offset:
        mag_offset = cosmo.distmod(z + 0.015).value - cosmo.distmod(z).value
        z += 0.015
    else:
        mag_offset = 0

    filters, filternames, flux_maggies, flux_maggies_error = [], [], [], []

    for i, cutout in enumerate(transient.host_phot_filters): #enumerate(transient.cutouts):
        filter_name = cutout['filter'].name  # Ensure filter name exists
        trans_curve = cutout['filter'].transmission_curve  # Ensure transmission data exists

        sedpy_filter = cutout['filter'].sedpy_name
        
        datapoint = photometry[i]

        if datapoint['flux'] is None:
            continue

        # kCorrect for MW reddening
        if aperture_type == "global":
            mwebv = transient.host.milkyway_dust_reddening
            if mwebv is None:
                mwebv = get_dust_maps(transient.host.sky_coord)
        elif aperture_type == "local":
            mwebv = transient.milkyway_dust_reddening
            if mwebv is None:
                mwebv = get_dust_maps(transient.sky_coord)
        else:
            raise ValueError(
                f"aperture_type must be 'global' or 'local', currently set to {aperture_type}"
            )

        wave_eff = cutout['filter'].wavelength_eff_angstrom  # filter effective wavelength
        ext_corr = extinction.fitzpatrick99(np.array([wave_eff]), mwebv * 3.1, r_v=3.1)[0]
        flux_mwcorr = (
            datapoint['flux'] * 10 ** (0.4 * ext_corr)
        )
        
        # 1% error floor
        fluxerr_mwcorr = np.sqrt(
            (
                datapoint['flux_error']
                * 10 ** (0.4 * ext_corr)
            ) ** 2.0
            + (0.01 * flux_mwcorr) ** 2.0
        )

        # TEST - are low-S/N observations messing up prospector?
        #if flux_mwcorr / fluxerr_mwcorr < 3:
        #    continue

        # Append the `sedpy` filter
        filternames.append(filter_name)
        filters.append(sedpy_filter)
        flux_maggies.append(mJy_to_maggies(flux_mwcorr * 10 ** (-0.4 * mag_offset)))
        flux_maggies_error.append(
            mJy_to_maggies(fluxerr_mwcorr * 10 ** (-0.4 * mag_offset))
        )
    obs_data = {}

    obs_data['wavelength'] = None
    obs_data['spectrum'] = None
    obs_data['unc'] = None
    obs_data['redshift'] = z
    obs_data['filters'] = load_filters(filters)
    obs_data["phot_wave"] = [f.wave_effective for f in obs_data["filters"]]
    obs_data['maggies'] =  np.array(flux_maggies)
    obs_data['maggies_unc'] = np.array(flux_maggies_error)
    obs_data['phot_mask'] = np.isfinite(np.squeeze(np.array(obs_data['maggies'])))
    obs_data['filternames'] = np.array(filternames)
    
    
    return obs_data


def build_model(observations):
    """
    Construct all model components
    """

    model = build_model_nonparam(observations)
    sps = FastStepBasis(zcontinuous=2, compute_vega_mags=False)
    noise_model = (None, None)
    return {"model": model, "sps": sps, "noise_model": noise_model}

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
        "prior": priors.FastUniform(a=0.0, b=1.5+ 1e-3)
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

def build_model_nonparam_OLD(obs=None, **extras):
    """prospector-alpha"""
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
        "gas_logz",
        "duste_qpah",
        "duste_umin",
        "log_duste_gamma",
    ]

    # -------------
    # MODEL_PARAMS
    model_params = {}

    # --- BASIC PARAMETERS ---
    model_params["zred"] = {
        "N": 1,
        "isfree": True,
        "init": 0.5,
        "prior": priors.FastUniform(a=0, b=0.2),
    }

    model_params["logmass"] = {
        "N": 1,
        "isfree": True,
        "init": 8.0,
        "units": "Msun",
        "prior": priors.FastUniform(a=7.0, b=12.5),
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
    model_params["add_neb_emission"] = {"N": 1, "isfree": False, "init": True}
    model_params["add_neb_continuum"] = {"N": 1, "isfree": False, "init": True}
    model_params["gas_logz"] = {
        "N": 1,
        "isfree": True,
        "init": -0.5,
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


