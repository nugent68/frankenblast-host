import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import corner
import prospect
from fit_host_sed import build_model,  build_model_nonparam
from postprocess_prosp import z_to_agebins, get_mwa
from prospect.sources import FastStepBasis
import prospect.io.read_results as reader
from prospect.utils.plotting import quantile
from prospect.plotting.utils import sample_posterior
from astropy.cosmology import WMAP9


 
def theta_posteriors(results, **kwargs):
    # Get parameter names 
    parnames = results['theta_labels']

     # Get the arrays we need (trace, wghts)
    theta_samp = results['chain']
    
    # convert dust2 to AV
    dust_AV = 1.086*total_dust(theta_samp[:, parnames.index('dust1_fraction')], 
                               theta_samp[:, parnames.index('dust2')])

    
    # nicer names
    
    mass_log = theta_samp[:, parnames.index('logmass')]
    logzsol = theta_samp[:, parnames.index('logzsol')]
    
    logSFR_1 = theta_samp[:, parnames.index('logsfr_ratios_1')]
    logSFR_2 = theta_samp[:, parnames.index('logsfr_ratios_2')]
    logSFR_3 = theta_samp[:, parnames.index('logsfr_ratios_3')]
    logSFR_4 = theta_samp[:, parnames.index('logsfr_ratios_4')]
    logSFR_5 = theta_samp[:, parnames.index('logsfr_ratios_5')]
    logSFR_6 = theta_samp[:, parnames.index('logsfr_ratios_6')]
    
    new_theta = []
    if 'zred' in parnames:
        zred = theta_samp[:, parnames.index('zred')]
        
        for i in np.arange(0, len(mass_log), 1):
            new_idx = [dust_AV[i], mass_log[i], logzsol[i], logSFR_1[i], logSFR_2[i], 
                       logSFR_3[i], logSFR_4[i], logSFR_5[i], logSFR_6[i], zred[i]]
            new_theta.append(new_idx)
    
    else:
        for i in np.arange(0, len(mass_log), 1):
            new_idx = [dust_AV[i], mass_log[i], logzsol[i], logSFR_1[i], logSFR_2[i], 
                       logSFR_3[i], logSFR_4[i], logSFR_5[i], logSFR_6[i]]
            new_theta.append(new_idx)
    
    new_theta_arr = np.array(new_theta)
    
    return new_theta_arr


# --------------
# SPS Object
# --------------
def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = FastStepBasis(zcontinuous=zcontinuous,
                        compute_vega_mags=compute_vega_mags)  # special to remove redshifting issue
    return sps

def total_dust(dust1_fraction, dust2):
    return dust1_fraction*dust2 + dust2



def get_posteriors(transient, file):

    
    # Load results
    run_params={}
    
    res, obs, _ = reader.results_from(file, dangerous=False)
    
    mod = build_model_nonparam(obs,**run_params)
    
    sps = build_sps(**run_params)

    host_all = {
    'AV': [],
    'redshift': [],
    'stellar mass': [],
    'logzsol':[],
    'age':[],
    'SFR':[],
    'sSFR':[]
    }

    
    if obs['redshift'] != None:
        zred = obs['redshift']
        host_all['redshift'].append(obs['redshift'])
    else:
        zred = np.median(theta_arr[:,-1])
        host_all['redshift'].append(theta_arr[:,-1])
   
    # Get posteriors
    theta_arr = theta_posteriors(res, truths=None)
    
    host_all['AV'].append(theta_arr[:,0])
    host_all['logzsol'].append(theta_arr[:,2])

    
    # Stellar mass calculation
    spec, phot, mfrac = mod.predict(mod.theta, obs=obs, sps=sps)
    
    
    npoints = res['chain'].shape[0]
    parnames = res['theta_labels']
    ncalc = len(res['chain'][:, parnames.index('logmass')]) # number of samples to generate
    thetas = res['chain']

    surviving_mass = []
    for theta in thetas:
        surviving_mass.append(np.sum(10**theta[parnames.index('logmass')]) * mfrac)
        
    host_all['stellar mass'].append(np.log10(np.array(surviving_mass)))
    
    
    # Star and Mass Formation History Plots
    nbins_sfh = 7
    

    agebins = z_to_agebins(zred=zred)
    resc = res['chain']


    SFR = []
    MASS = []
    sfr = np.zeros((npoints, nbins_sfh)) # change to number of bins
    mass = np.zeros((npoints, nbins_sfh)) # change to number of bins


    for sample in range(len(res['chain'][:, parnames.index('logmass')])):
        sfr[sample] = prospect.models.transforms.logsfr_ratios_to_sfrs(logmass=resc[sample][parnames.index('logmass')],
                                                                       logsfr_ratios=[resc[sample][parnames.index('logsfr_ratios_1')],
                                                                                      resc[sample][parnames.index('logsfr_ratios_2')],
                                                                                      resc[sample][parnames.index('logsfr_ratios_3')],
                                                                                      resc[sample][parnames.index('logsfr_ratios_4')],
                                                                                      resc[sample][parnames.index('logsfr_ratios_5')],
                                                                                      resc[sample][parnames.index('logsfr_ratios_6')]               
                                                                                     ],
                                                                       
                                                                       agebins=agebins)
        SFR.append(sfr[sample])
        mass[sample] = prospect.models.transforms.logsfr_ratios_to_masses(logmass=resc[sample][parnames.index('logmass')],
                                                                          logsfr_ratios=[resc[sample][parnames.index('logsfr_ratios_1')],
                                                                                      resc[sample][parnames.index('logsfr_ratios_2')],
                                                                                      resc[sample][parnames.index('logsfr_ratios_3')],
                                                                                      resc[sample][parnames.index('logsfr_ratios_4')],
                                                                                      resc[sample][parnames.index('logsfr_ratios_5')],
                                                                                      resc[sample][parnames.index('logsfr_ratios_6')]               
                                                                                     ],
                                                                       
                                                                       agebins=agebins)
        MASS.append(mass[sample])
        
        
    # Now we have a numpy array of the mass and SFR in each bin
    star_formation = [list(j) for j in SFR]
    s_mass = [list(j) for j in MASS]
    npSFR = np.array(star_formation, dtype=float)
    npSM = np.array(s_mass, dtype=float)

    #Calculate the quantiles
    Q16_SFR = [quantile(npSFR[:, i], percents=16, weights = res.get("WEIGHTS", None))
                 for i in range(nbins_sfh)]
             
    Q50_SFR = [quantile(npSFR[:, i], percents=50, weights = res.get("WEIGHTS", None))
                 for i in range(nbins_sfh)]

    Q84_SFR = [quantile(npSFR[:, i], percents=84, weights=res.get("WEIGHTS", None))
                 for i in range(nbins_sfh)]      

    Q16_Mass = [quantile(npSM[:, i], percents=16, weights = res.get("WEIGHTS", None))
                 for i in range(nbins_sfh)]

    Q50_Mass = [quantile(npSM[:, i], percents=50, weights = res.get("WEIGHTS", None))
                 for i in range(nbins_sfh)]

    Q84_Mass = [quantile(npSM[:, i], percents=84, weights = res.get("WEIGHTS", None))
                 for i in range(nbins_sfh)]
    
    
    
    # sSFR and Type SF Calc
    sfr_theta_samp = sample_posterior(npSFR, weights=res.get("WEIGHTS", None), nsample=100000)
    
    #host_all['sSFR'].append(ssfr_Arr)
    host_all['SFR'].append(sfr_theta_samp[:,0])
                           
    MWA_arr = []
    # Age Calculation
    for p in npSFR:
        MWA_arr.append(get_mwa(agebins, p)) 
    
    host_all['age'].append(MWA_arr)

    # Save dictionary
    outfile = './data/sed_output/' + transient.name + '/' + transient.name +'sp_prop.npy'
    
    np.save(outfile, host_all)   

    return host_all