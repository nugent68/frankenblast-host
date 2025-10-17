#!/usr/bin/env python
"""
FrankenBlast Simplified Wrapper

Provides simplified functions for common FrankenBlast operations.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

# FrankenBlast imports
from classes import Transient as BlastTransient, Host, Filter
from get_host import run_prost
from get_host_images import download_and_save_cutouts, get_cutouts, survey_list
from create_apertures import construct_aperture
from do_photometry import do_global_photometry
from fit_host_sed import fit_host
from mwebv_host import get_mwebv
import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FrankenBlast')


class FrankenBlast:
    """Simplified FrankenBlast interface for Python scripts"""
    
    def __init__(self, name: str, ra: float, dec: float, redshift: Optional[float] = None):
        """
        Initialize FrankenBlast for a transient.
        
        Parameters
        ----------
        name : str
            Transient name
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        redshift : float, optional
            Transient redshift if known
        """
        self.name = name
        self.ra = ra
        self.dec = dec
        self.redshift = redshift
        
        # Create transient object
        coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        self.transient = BlastTransient(
            name=name,
            coordinates=coords,
            transient_redshift=redshift
        )
        
        # Storage for results
        self.host_info = None
        self.photometry = None
        self.sed_result = None
        
        logger.info(f"Initialized FrankenBlast for {name} at RA={ra:.6f}, Dec={dec:.6f}")
    
    def find_host(self, 
                  n_samples: int = 1000,
                  catalogs: Optional[List[str]] = None,
                  save_output: bool = False,
                  output_dir: str = 'prostdb') -> Dict:
        """
        Find the host galaxy using PROST.
        
        Parameters
        ----------
        n_samples : int
            Number of Monte Carlo samples for PROST
        catalogs : list, optional
            List of catalogs to query. Default: ['panstarrs', 'glade', 'decals']
        save_output : bool
            Whether to save PROST output to file
        output_dir : str
            Directory for saving PROST output
            
        Returns
        -------
        dict
            Host association results
        """
        logger.info("Starting host association...")
        
        if catalogs is None:
            catalogs = ['glade', 'decals']  # Excluding panstarrs due to known issue
        
        # Run PROST
        self.host_info = run_prost(
            self.transient,
            output_dir=output_dir,
            save=save_output
        )
        
        # Set host if found
        if self.host_info['host_ra'] is not None:
            host_coords = SkyCoord(
                ra=self.host_info['host_ra']*u.deg,
                dec=self.host_info['host_dec']*u.deg
            )
            self.transient.host = Host(
                sky_coord=host_coords,
                redshift=self.host_info['host_redshift_mean']
            )
            
            logger.info(f"Found host at RA={self.host_info['host_ra']:.6f}, "
                       f"Dec={self.host_info['host_dec']:.6f}")
            logger.info(f"Host redshift: {self.host_info['host_redshift_mean']:.4f} "
                       f"± {self.host_info['host_redshift_std']:.4f}")
            logger.info(f"Host probability: {self.host_info['host_prob']:.3f}")
        else:
            logger.warning("No host galaxy found")
            
        return self.host_info
    
    def set_host(self, ra: float, dec: float, redshift: Optional[float] = None):
        """
        Manually set the host galaxy.
        
        Parameters
        ----------
        ra : float
            Host RA in degrees
        dec : float
            Host Dec in degrees
        redshift : float, optional
            Host redshift
        """
        host_coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        self.transient.host = Host(
            sky_coord=host_coords,
            redshift=redshift
        )
        logger.info(f"Set host at RA={ra:.6f}, Dec={dec:.6f}, z={redshift}")
    
    def download_images(self, 
                       surveys: Optional[List[str]] = None,
                       size: float = 120.0) -> int:
        """
        Download cutout images from surveys.
        
        Parameters
        ----------
        surveys : list, optional
            List of surveys to query. Default: all available
        size : float
            Cutout size in arcseconds
            
        Returns
        -------
        int
            Number of cutouts downloaded
        """
        if surveys is None:
            surveys = ['PanSTARRS', 'DES', 'WISE', '2MASS', 'GALEX']
        
        logger.info(f"Downloading cutouts from {len(surveys)} surveys...")
        
        for survey in surveys:
            try:
                logger.info(f"Querying {survey}...")
                download_and_save_cutouts(
                    self.transient.name,
                    self.transient.coordinates.ra.deg,
                    self.transient.coordinates.dec.deg,
                    survey,
                    size=size
                )
            except Exception as e:
                logger.warning(f"Failed to download {survey} cutouts: {e}")
        
        # Load cutouts
        self.transient.cutouts = get_cutouts(
            self.transient.name, 
            self.transient.coordinates
        )
        
        logger.info(f"Downloaded {len(self.transient.cutouts)} cutouts")
        return len(self.transient.cutouts)
    
    def do_photometry(self, 
                     aperture_type: str = 'global',
                     fwhm_correction: bool = True,
                     show_plots: bool = False,
                     save_results: bool = True,
                     output_file: Optional[str] = None) -> Dict:
        """
        Perform aperture photometry on the host.
        
        Parameters
        ----------
        aperture_type : str
            'global' or 'local' aperture photometry
        fwhm_correction : bool
            Apply FWHM correction to apertures
        show_plots : bool
            Display photometry plots
        save_results : bool
            Save photometry results to file
        output_file : str, optional
            Path for output file. Default: {name}_photometry.json
            
        Returns
        -------
        dict
            Photometry results
        """
        if not self.transient.host:
            raise ValueError("No host set. Run find_host() or set_host() first.")
        
        # Ensure we have cutouts
        if not hasattr(self.transient, 'cutouts') or not self.transient.cutouts:
            self.download_images()
        
        logger.info("Starting photometry...")
        
        # Construct apertures
        logger.info("Constructing apertures...")
        self.transient.global_apertures = []
        
        for cutout in self.transient.cutouts:
            try:
                aperture = construct_aperture(
                    cutout,
                    self.transient,
                    show_plot=show_plots
                )
                self.transient.global_apertures.append(aperture)
            except Exception as e:
                logger.warning(f"Failed to create aperture for {cutout['filter'].name}: {e}")
                self.transient.global_apertures.append(None)
        
        # Perform photometry
        logger.info("Performing aperture photometry...")
        photometry_results = []
        
        for i, cutout in enumerate(self.transient.cutouts):
            if self.transient.global_apertures[i] is not None:
                try:
                    phot = do_global_photometry(
                        self.transient,
                        filter=cutout['filter'],
                        aperture=self.transient.global_apertures[i],
                        fwhm_correction=fwhm_correction,
                        show_plot=show_plots
                    )
                    photometry_results.append(phot)
                    
                    if phot['flux'] is not None:
                        logger.info(f"{cutout['filter'].name}: "
                                  f"{phot['magnitude']:.2f} ± {phot['magnitude_error']:.2f} mag")
                except Exception as e:
                    logger.warning(f"Photometry failed for {cutout['filter'].name}: {e}")
                    photometry_results.append({
                        'flux': None, 'flux_error': None,
                        'magnitude': None, 'magnitude_error': None
                    })
            else:
                photometry_results.append({
                    'flux': None, 'flux_error': None,
                    'magnitude': None, 'magnitude_error': None
                })
        
        self.transient.host_photometry = photometry_results
        self.transient.host_phot_filters = self.transient.cutouts
        
        # Create summary
        self.photometry = {
            'transient_name': self.name,
            'transient_ra': self.ra,
            'transient_dec': self.dec,
            'host_ra': self.transient.host.sky_coord.ra.deg,
            'host_dec': self.transient.host.sky_coord.dec.deg,
            'host_redshift': self.transient.host.redshift,
            'filters': []
        }
        
        valid_count = 0
        for i, phot in enumerate(photometry_results):
            filter_data = {
                'name': self.transient.cutouts[i]['filter'].name,
                'wavelength_eff': self.transient.cutouts[i]['filter'].wavelength_eff_angstrom,
                'flux_mJy': phot['flux'],
                'flux_error_mJy': phot['flux_error'],
                'magnitude': phot['magnitude'],
                'magnitude_error': phot['magnitude_error']
            }
            self.photometry['filters'].append(filter_data)
            if phot['flux'] is not None:
                valid_count += 1
        
        logger.info(f"Photometry complete: {valid_count}/{len(photometry_results)} valid measurements")
        
        # Save results
        if save_results:
            if output_file is None:
                output_file = f"{self.name}_photometry.json"
            
            with open(output_file, 'w') as f:
                json.dump(self.photometry, f, indent=2)
            logger.info(f"Saved photometry to {output_file}")
        
        return self.photometry
    
    def fit_sed(self,
               use_sbipp: bool = True,
               model: str = 'zfree_GPD2W',
               mode: str = 'test',
               save_results: bool = True) -> str:
        """
        Fit the host galaxy SED.
        
        Parameters
        ----------
        use_sbipp : bool
            Use SBI++ for faster fitting
        model : str
            SBI model: 'zfree_GPD2W' or 'zfix_GPD2W'
        mode : str
            'test' or 'production' mode
        save_results : bool
            Save SED fitting results
            
        Returns
        -------
        str
            Status message
        """
        if not self.transient.host_photometry:
            raise ValueError("No photometry available. Run do_photometry() first.")
        
        logger.info("Starting SED fitting...")
        
        # Check for required files
        if model == 'zfree_GPD2W':
            fname = 'zfree_GPD2W'
            model_file = 'data/SBI/SBI_model_zfree_GPD2W_global.pt'
            summary_file = 'data/SBI/SBI_zfree_GPD2W_model_summary.p'
        else:
            fname = 'zfix_GPD2W'
            model_file = 'data/SBI/SBI_model_zfix_GPD2W_global.pt'
            summary_file = 'data/SBI/SBI_zfix_GPD2W_model_summary.p'
        
        if not os.path.exists(model_file):
            logger.error(f"SBI model file not found: {model_file}")
            logger.error("Please download the SBI models from Zenodo (see README)")
            return "Model files not found"
        
        # Load SBI parameters
        import pickle
        import torch
        
        with open(summary_file, 'rb') as f:
            model_summary = pickle.load(f)
        
        sbi_params = {
            'device': 'cpu',
            'model': torch.load(model_file, map_location='cpu'),
            'model_summary': model_summary
        }
        
        # Get all filters
        all_filters = []
        for survey in ['PanSTARRS', 'DES', 'WISE', '2MASS', 'GALEX']:
            filters = survey_list(survey, self.transient.coordinates)
            all_filters.extend(filters)
        
        # Run fitting
        try:
            self.sed_result = fit_host(
                self.transient,
                sbi_params=sbi_params,
                fname=fname,
                all_filters=all_filters,
                mode=mode,
                sbipp=use_sbipp,
                aperture_type='global',
                save=save_results
            )
            
            if self.sed_result == 'done':
                logger.info("SED fitting completed successfully")
                output_path = f"data/sed_output/{self.name}/{self.name}_global.h5"
                logger.info(f"Results saved to {output_path}")
            else:
                logger.warning(f"SED fitting issue: {self.sed_result}")
                
        except Exception as e:
            logger.error(f"SED fitting failed: {e}")
            self.sed_result = f"Error: {e}"
            
        return self.sed_result
    
    def run_full_pipeline(self,
                         find_host_kwargs: Optional[Dict] = None,
                         photometry_kwargs: Optional[Dict] = None,
                         sed_kwargs: Optional[Dict] = None) -> Dict:
        """
        Run the complete FrankenBlast pipeline.
        
        Parameters
        ----------
        find_host_kwargs : dict, optional
            Arguments for find_host()
        photometry_kwargs : dict, optional
            Arguments for do_photometry()
        sed_kwargs : dict, optional
            Arguments for fit_sed()
            
        Returns
        -------
        dict
            Summary of pipeline results
        """
        logger.info("="*60)
        logger.info(f"Running full FrankenBlast pipeline for {self.name}")
        logger.info("="*60)
        
        # Default kwargs
        if find_host_kwargs is None:
            find_host_kwargs = {}
        if photometry_kwargs is None:
            photometry_kwargs = {}
        if sed_kwargs is None:
            sed_kwargs = {}
        
        # Step 1: Find host
        logger.info("\nStep 1: Finding host galaxy...")
        host_info = self.find_host(**find_host_kwargs)
        
        if not self.transient.host:
            logger.error("No host found. Pipeline stopped.")
            return {
                'success': False,
                'host_found': False,
                'message': 'No host galaxy found'
            }
        
        # Step 2: Download images
        logger.info("\nStep 2: Downloading cutout images...")
        n_cutouts = self.download_images()
        
        # Step 3: Photometry
        logger.info("\nStep 3: Performing photometry...")
        photometry = self.do_photometry(**photometry_kwargs)
        
        # Step 4: SED fitting
        logger.info("\nStep 4: Fitting SED...")
        sed_result = self.fit_sed(**sed_kwargs)
        
        # Summary
        summary = {
            'success': True,
            'transient_name': self.name,
            'transient_ra': self.ra,
            'transient_dec': self.dec,
            'transient_redshift': self.redshift,
            'host_found': True,
            'host_ra': self.transient.host.sky_coord.ra.deg,
            'host_dec': self.transient.host.sky_coord.dec.deg,
            'host_redshift': self.transient.host.redshift,
            'host_probability': host_info['host_prob'],
            'n_cutouts': n_cutouts,
            'n_photometry': sum(1 for f in photometry['filters'] if f['flux_mJy'] is not None),
            'sed_status': sed_result
        }
        
        logger.info("\n" + "="*60)
        logger.info("Pipeline completed!")
        logger.info("="*60)
        logger.info(f"\nSummary:")
        logger.info(f"  Host found: Yes")
        logger.info(f"  Host redshift: {summary['host_redshift']:.4f}")
        logger.info(f"  Host probability: {summary['host_probability']:.3f}")
        logger.info(f"  Valid photometry: {summary['n_photometry']}/{n_cutouts}")
        logger.info(f"  SED fitting: {summary['sed_status']}")
        
        return summary


# Convenience functions for quick analysis
def quick_host_association(name: str, ra: float, dec: float, 
                          redshift: Optional[float] = None) -> Dict:
    """
    Quick host association for a transient.
    
    Parameters
    ----------
    name : str
        Transient name
    ra : float
        RA in degrees
    dec : float
        Dec in degrees
    redshift : float, optional
        Transient redshift
        
    Returns
    -------
    dict
        Host association results
    """
    fb = FrankenBlast(name, ra, dec, redshift)
    return fb.find_host()


def quick_photometry(name: str, ra: float, dec: float,
                    host_ra: float, host_dec: float,
                    host_redshift: Optional[float] = None) -> Dict:
    """
    Quick photometry for a known host.
    
    Parameters
    ----------
    name : str
        Transient name
    ra : float
        Transient RA in degrees
    dec : float
        Transient Dec in degrees
    host_ra : float
        Host RA in degrees
    host_dec : float
        Host Dec in degrees
    host_redshift : float, optional
        Host redshift
        
    Returns
    -------
    dict
        Photometry results
    """
    fb = FrankenBlast(name, ra, dec)
    fb.set_host(host_ra, host_dec, host_redshift)
    fb.download_images()
    return fb.do_photometry()


def quick_sed_fit(name: str, ra: float, dec: float,
                 photometry_file: str) -> str:
    """
    Quick SED fitting from saved photometry.
    
    Parameters
    ----------
    name : str
        Transient name
    ra : float
        RA in degrees
    dec : float
        Dec in degrees
    photometry_file : str
        Path to photometry JSON file
        
    Returns
    -------
    str
        SED fitting status
    """
    # Load photometry
    with open(photometry_file, 'r') as f:
        phot_data = json.load(f)
    
    # Create FrankenBlast instance
    fb = FrankenBlast(name, ra, dec)
    
    # Set host
    fb.set_host(
        phot_data['host_ra'],
        phot_data['host_dec'],
        phot_data.get('host_redshift')
    )
    
    # Load photometry into transient
    fb.transient.host_photometry = []
    fb.transient.host_phot_filters = []
    
    for filt in phot_data['filters']:
        fb.transient.host_photometry.append({
            'flux': filt['flux_mJy'],
            'flux_error': filt['flux_error_mJy'],
            'magnitude': filt['magnitude'],
            'magnitude_error': filt['magnitude_error']
        })
        
        # Create minimal filter object
        filter_obj = type('Filter', (), {
            'name': filt['name'],
            'wavelength_eff_angstrom': filt['wavelength_eff']
        })()
        fb.transient.host_phot_filters.append({'filter': filter_obj})
    
    # Run SED fitting
    return fb.fit_sed()


if __name__ == '__main__':
    # Example usage
    print("FrankenBlast wrapper loaded. Example usage:")
    print()
    print("from frankenblast import FrankenBlast")
    print()
    print("# Initialize for a transient")
    print("fb = FrankenBlast('SN2019ulo', ra=20.509, dec=-60.571)")
    print()
    print("# Run full pipeline")
    print("results = fb.run_full_pipeline()")
    print()
    print("# Or run individual steps")
    print("fb.find_host()")
    print("fb.download_images()")
    print("fb.do_photometry()")
    print("fb.fit_sed()")