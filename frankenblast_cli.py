#!/usr/bin/env python
"""
FrankenBlast Command Line Interface

A unified CLI for transient host association, photometry, and SED fitting.
"""

import argparse
import sys
import os
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

# FrankenBlast imports
from classes import Transient as BlastTransient, Host, Filter
from get_host import run_prost
from get_host_images import download_and_save_cutouts, get_cutouts
from create_apertures import construct_aperture
from do_photometry import do_global_photometry
from fit_host_sed import fit_host
from mwebv_host import get_mwebv
import settings

# Configure logging
def setup_logging(verbose=False, log_file=None):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )
    return logging.getLogger('FrankenBlast')

class FrankenBlastCLI:
    """Main CLI handler for FrankenBlast"""
    
    def __init__(self):
        self.parser = self.create_parser()
        self.logger = None
        
    def create_parser(self):
        """Create the argument parser with subcommands"""
        parser = argparse.ArgumentParser(
            description='FrankenBlast: Rapid transient host association, photometry, and SED fitting',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run full pipeline for a transient
  frankenblast_cli.py full --name SN2019ulo --ra 20.509 --dec -60.571 --config config.yaml
  
  # Just find host association
  frankenblast_cli.py host --name SN2019ulo --ra 20.509 --dec -60.571
  
  # Run photometry on existing host
  frankenblast_cli.py photometry --name SN2019ulo --host-ra 20.509 --host-dec -60.571
  
  # Fit SED for host with photometry
  frankenblast_cli.py sed --name SN2019ulo --photometry-file photometry.json
            """
        )
        
        # Global arguments
        parser.add_argument('-v', '--verbose', action='store_true',
                          help='Enable verbose output')
        parser.add_argument('--log-file', type=str,
                          help='Path to log file')
        parser.add_argument('--config', type=str,
                          help='Path to configuration file (YAML or JSON)')
        
        # Create subparsers
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Full pipeline command
        full_parser = subparsers.add_parser('full', 
                                           help='Run full pipeline: host association, photometry, and SED fitting')
        self.add_transient_args(full_parser)
        self.add_host_args(full_parser)
        self.add_photometry_args(full_parser)
        self.add_sed_args(full_parser)
        
        # Host association command
        host_parser = subparsers.add_parser('host', 
                                           help='Find host galaxy association')
        self.add_transient_args(host_parser)
        self.add_host_args(host_parser)
        
        # Photometry command
        phot_parser = subparsers.add_parser('photometry', 
                                           help='Perform host photometry')
        self.add_transient_args(phot_parser)
        self.add_photometry_args(phot_parser)
        phot_parser.add_argument('--host-ra', type=float,
                                help='Host RA (degrees)')
        phot_parser.add_argument('--host-dec', type=float,
                                help='Host Dec (degrees)')
        phot_parser.add_argument('--host-redshift', type=float,
                                help='Host redshift')
        
        # SED fitting command
        sed_parser = subparsers.add_parser('sed', 
                                          help='Fit host SED')
        self.add_transient_args(sed_parser)
        self.add_sed_args(sed_parser)
        sed_parser.add_argument('--photometry-file', type=str,
                               help='Path to photometry results file')
        
        # Cutouts command
        cutout_parser = subparsers.add_parser('cutouts', 
                                             help='Download cutout images')
        self.add_transient_args(cutout_parser)
        cutout_parser.add_argument('--surveys', nargs='+', 
                                  default=['PanSTARRS', 'DES', 'WISE', '2MASS', 'GALEX'],
                                  help='Surveys to query')
        cutout_parser.add_argument('--size', type=float, default=120.0,
                                  help='Cutout size in arcseconds')
        
        return parser
    
    def add_transient_args(self, parser):
        """Add transient-specific arguments"""
        parser.add_argument('--name', type=str, required=True,
                          help='Transient name')
        parser.add_argument('--ra', type=float,
                          help='Transient RA (degrees)')
        parser.add_argument('--dec', type=float,
                          help='Transient Dec (degrees)')
        parser.add_argument('--redshift', type=float,
                          help='Transient redshift (optional)')
        
    def add_host_args(self, parser):
        """Add host association arguments"""
        parser.add_argument('--prost-samples', type=int, default=1000,
                          help='Number of PROST MC samples')
        parser.add_argument('--prost-catalogs', nargs='+',
                          default=['panstarrs', 'glade', 'decals'],
                          help='Catalogs for PROST to query')
        parser.add_argument('--save-prost', action='store_true',
                          help='Save PROST output to file')
        parser.add_argument('--prost-output-dir', type=str, default='prostdb',
                          help='Directory for PROST output')
        
    def add_photometry_args(self, parser):
        """Add photometry arguments"""
        parser.add_argument('--aperture-type', choices=['global', 'local'], 
                          default='global',
                          help='Type of aperture photometry')
        parser.add_argument('--fwhm-correction', action='store_true', default=True,
                          help='Apply FWHM correction to apertures')
        parser.add_argument('--show-plots', action='store_true',
                          help='Display photometry plots')
        parser.add_argument('--photometry-output', type=str,
                          help='Output file for photometry results')
        
    def add_sed_args(self, parser):
        """Add SED fitting arguments"""
        parser.add_argument('--use-sbipp', action='store_true', default=True,
                          help='Use SBI++ for SED fitting (faster)')
        parser.add_argument('--sed-mode', choices=['test', 'production'], 
                          default='test',
                          help='SED fitting mode')
        parser.add_argument('--sed-output-dir', type=str, 
                          default='data/sed_output',
                          help='Directory for SED output')
        parser.add_argument('--sbi-model', type=str, 
                          default='zfree_GPD2W',
                          help='SBI model to use (zfree_GPD2W or zfix_GPD2W)')
        
    def load_config(self, config_path):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            elif config_path.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError("Config file must be YAML or JSON")
    
    def merge_args_with_config(self, args):
        """Merge command line args with config file"""
        if args.config:
            config = self.load_config(args.config)
            # Command line args override config file
            for key, value in config.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
        return args
    
    def create_transient(self, args):
        """Create a Transient object from arguments"""
        if not args.ra or not args.dec:
            raise ValueError("RA and Dec are required for transient")
        
        coords = SkyCoord(ra=args.ra*u.deg, dec=args.dec*u.deg)
        
        transient = BlastTransient(
            name=args.name,
            coordinates=coords,
            transient_redshift=getattr(args, 'redshift', None)
        )
        
        self.logger.info(f"Created transient: {args.name} at RA={args.ra}, Dec={args.dec}")
        return transient
    
    def run_host_association(self, transient, args):
        """Run PROST host association"""
        self.logger.info("Starting host association with PROST...")
        
        # Configure catalogs
        catalogs = {}
        for cat in args.prost_catalogs:
            if cat == 'panstarrs':
                catalogs['panstarrs'] = 'dr2'
            elif cat == 'glade':
                catalogs['glade'] = 'latest'
            elif cat == 'decals':
                catalogs['decals'] = 'dr10'
        
        # Run PROST
        hosts = run_prost(
            transient, 
            output_dir=args.prost_output_dir,
            save=args.save_prost
        )
        
        if hosts['host_ra'] is not None:
            # Create Host object
            host_coords = SkyCoord(ra=hosts['host_ra']*u.deg, 
                                 dec=hosts['host_dec']*u.deg)
            host = Host(
                sky_coord=host_coords,
                redshift=hosts['host_redshift_mean']
            )
            transient.host = host
            
            self.logger.info(f"Found host at RA={hosts['host_ra']:.6f}, Dec={hosts['host_dec']:.6f}")
            self.logger.info(f"Host redshift: {hosts['host_redshift_mean']:.4f} ± {hosts['host_redshift_std']:.4f}")
            self.logger.info(f"Host probability: {hosts['host_prob']:.3f}")
        else:
            self.logger.warning("No host galaxy found")
            
        return hosts
    
    def download_cutouts(self, transient, args):
        """Download cutout images"""
        self.logger.info("Downloading cutout images...")
        
        surveys = getattr(args, 'surveys', ['PanSTARRS', 'DES', 'WISE', '2MASS', 'GALEX'])
        size = getattr(args, 'size', 120.0)
        
        for survey in surveys:
            try:
                self.logger.info(f"Querying {survey}...")
                download_and_save_cutouts(
                    transient.name,
                    transient.coordinates.ra.deg,
                    transient.coordinates.dec.deg,
                    survey,
                    size=size
                )
            except Exception as e:
                self.logger.warning(f"Failed to download {survey} cutouts: {e}")
        
        # Load cutouts into transient
        transient.cutouts = get_cutouts(transient.name, transient.coordinates)
        self.logger.info(f"Downloaded {len(transient.cutouts)} cutouts")
        
    def run_photometry(self, transient, args):
        """Run aperture photometry"""
        self.logger.info("Starting photometry...")
        
        # Ensure we have cutouts
        if not hasattr(transient, 'cutouts') or not transient.cutouts:
            self.download_cutouts(transient, args)
        
        # Construct apertures
        self.logger.info("Constructing apertures...")
        for cutout in transient.cutouts:
            try:
                aperture = construct_aperture(
                    cutout, 
                    transient,
                    show_plot=args.show_plots
                )
                transient.global_apertures.append(aperture)
            except Exception as e:
                self.logger.warning(f"Failed to create aperture for {cutout['filter'].name}: {e}")
                transient.global_apertures.append(None)
        
        # Perform photometry
        self.logger.info("Performing aperture photometry...")
        photometry_results = []
        
        for i, cutout in enumerate(transient.cutouts):
            if transient.global_apertures[i] is not None:
                try:
                    phot = do_global_photometry(
                        transient,
                        filter=cutout['filter'],
                        aperture=transient.global_apertures[i],
                        fwhm_correction=args.fwhm_correction,
                        show_plot=args.show_plots
                    )
                    photometry_results.append(phot)
                    
                    if phot['flux'] is not None:
                        self.logger.info(f"{cutout['filter'].name}: {phot['magnitude']:.2f} ± {phot['magnitude_error']:.2f} mag")
                    else:
                        self.logger.warning(f"{cutout['filter'].name}: No valid photometry")
                except Exception as e:
                    self.logger.warning(f"Photometry failed for {cutout['filter'].name}: {e}")
                    photometry_results.append({
                        'flux': None, 'flux_error': None,
                        'magnitude': None, 'magnitude_error': None
                    })
            else:
                photometry_results.append({
                    'flux': None, 'flux_error': None,
                    'magnitude': None, 'magnitude_error': None
                })
        
        transient.host_photometry = photometry_results
        transient.host_phot_filters = transient.cutouts
        
        # Save photometry results if requested
        if args.photometry_output:
            self.save_photometry(transient, args.photometry_output)
            
        return photometry_results
    
    def save_photometry(self, transient, output_file):
        """Save photometry results to file"""
        results = {
            'transient_name': transient.name,
            'transient_ra': transient.coordinates.ra.deg,
            'transient_dec': transient.coordinates.dec.deg,
            'transient_redshift': transient.transient_redshift,
            'photometry': []
        }
        
        if hasattr(transient, 'host') and transient.host:
            results['host_ra'] = transient.host.sky_coord.ra.deg
            results['host_dec'] = transient.host.sky_coord.dec.deg
            results['host_redshift'] = transient.host.redshift
        
        for i, phot in enumerate(transient.host_photometry):
            filter_info = {
                'filter': transient.cutouts[i]['filter'].name,
                'wavelength_eff': transient.cutouts[i]['filter'].wavelength_eff_angstrom,
                'flux_mJy': phot['flux'],
                'flux_error_mJy': phot['flux_error'],
                'magnitude': phot['magnitude'],
                'magnitude_error': phot['magnitude_error']
            }
            results['photometry'].append(filter_info)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved photometry to {output_file}")
    
    def load_photometry(self, transient, photometry_file):
        """Load photometry from file"""
        with open(photometry_file, 'r') as f:
            data = json.load(f)
        
        # Create filter objects and photometry
        transient.host_photometry = []
        transient.host_phot_filters = []
        
        for phot in data['photometry']:
            # Create basic photometry dict
            transient.host_photometry.append({
                'flux': phot['flux_mJy'],
                'flux_error': phot['flux_error_mJy'],
                'magnitude': phot['magnitude'],
                'magnitude_error': phot['magnitude_error']
            })
            
            # Create filter placeholder
            filter_obj = type('Filter', (), {
                'name': phot['filter'],
                'wavelength_eff_angstrom': phot['wavelength_eff']
            })()
            transient.host_phot_filters.append({'filter': filter_obj})
        
        # Set host info if available
        if 'host_ra' in data:
            host_coords = SkyCoord(ra=data['host_ra']*u.deg, 
                                 dec=data['host_dec']*u.deg)
            transient.host = Host(
                sky_coord=host_coords,
                redshift=data.get('host_redshift')
            )
        
        self.logger.info(f"Loaded photometry from {photometry_file}")
    
    def run_sed_fitting(self, transient, args):
        """Run SED fitting"""
        self.logger.info("Starting SED fitting...")
        
        # Load SBI model parameters
        if args.sbi_model == 'zfree_GPD2W':
            fname = 'zfree_GPD2W'
            model_file = 'data/SBI/SBI_model_zfree_GPD2W_global.pt'
            summary_file = 'data/SBI/SBI_zfree_GPD2W_model_summary.p'
        else:
            fname = 'zfix_GPD2W'
            model_file = 'data/SBI/SBI_model_zfix_GPD2W_global.pt'
            summary_file = 'data/SBI/SBI_zfix_GPD2W_model_summary.p'
        
        # Check if model files exist
        if not os.path.exists(model_file):
            self.logger.error(f"SBI model file not found: {model_file}")
            self.logger.error("Please download the SBI models from Zenodo (see README)")
            return None
        
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
        from get_host_images import survey_list
        all_filters = []
        for survey in ['PanSTARRS', 'DES', 'WISE', '2MASS', 'GALEX']:
            filters = survey_list(survey, transient.coordinates)
            all_filters.extend(filters)
        
        # Run SED fitting
        try:
            result = fit_host(
                transient,
                sbi_params=sbi_params,
                fname=fname,
                all_filters=all_filters,
                mode=args.sed_mode,
                sbipp=args.use_sbipp,
                aperture_type=args.aperture_type,
                save=True
            )
            
            if result == 'done':
                self.logger.info("SED fitting completed successfully")
                output_path = f"{args.sed_output_dir}/{transient.name}/{transient.name}_global.h5"
                self.logger.info(f"Results saved to {output_path}")
            else:
                self.logger.warning(f"SED fitting issue: {result}")
                
        except Exception as e:
            self.logger.error(f"SED fitting failed: {e}")
            return None
        
        return result
    
    def run_full_pipeline(self, args):
        """Run the complete pipeline"""
        self.logger.info("="*60)
        self.logger.info("Running full FrankenBlast pipeline")
        self.logger.info("="*60)
        
        # Create transient
        transient = self.create_transient(args)
        
        # Step 1: Host association
        self.logger.info("\n" + "="*40)
        self.logger.info("STEP 1: Host Association")
        self.logger.info("="*40)
        hosts = self.run_host_association(transient, args)
        
        if transient.host is None:
            self.logger.error("No host found. Cannot continue with photometry.")
            return
        
        # Step 2: Download cutouts
        self.logger.info("\n" + "="*40)
        self.logger.info("STEP 2: Downloading Cutouts")
        self.logger.info("="*40)
        self.download_cutouts(transient, args)
        
        # Step 3: Photometry
        self.logger.info("\n" + "="*40)
        self.logger.info("STEP 3: Photometry")
        self.logger.info("="*40)
        photometry = self.run_photometry(transient, args)
        
        # Step 4: SED fitting
        self.logger.info("\n" + "="*40)
        self.logger.info("STEP 4: SED Fitting")
        self.logger.info("="*40)
        sed_result = self.run_sed_fitting(transient, args)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Pipeline completed successfully!")
        self.logger.info("="*60)
        
        # Summary
        self.logger.info("\nSUMMARY:")
        self.logger.info(f"  Transient: {transient.name}")
        self.logger.info(f"  Host found: {'Yes' if transient.host else 'No'}")
        if transient.host:
            self.logger.info(f"  Host redshift: {transient.host.redshift:.4f}")
        self.logger.info(f"  Photometry points: {sum(1 for p in photometry if p['flux'] is not None)}/{len(photometry)}")
        self.logger.info(f"  SED fitting: {'Success' if sed_result == 'done' else 'Failed'}")
    
    def run(self):
        """Main entry point"""
        args = self.parser.parse_args()
        
        # Setup logging
        self.logger = setup_logging(args.verbose, args.log_file)
        
        # Check for required environment variables
        required_env = ['PROST_PATH', 'SBIPP_ROOT', 'SBIPP_PHOT_ROOT', 
                        'SBIPP_TRAINING_ROOT', 'SED_OUTPUT_ROOT']
        missing_env = [env for env in required_env if not os.environ.get(env)]
        if missing_env:
            self.logger.error(f"Missing required environment variables: {missing_env}")
            self.logger.error("Please source settings.sh or set these variables manually")
            sys.exit(1)
        
        # Merge config file with command line args
        args = self.merge_args_with_config(args)
        
        # Route to appropriate command
        if args.command == 'full':
            self.run_full_pipeline(args)
            
        elif args.command == 'host':
            transient = self.create_transient(args)
            hosts = self.run_host_association(transient, args)
            
        elif args.command == 'photometry':
            transient = self.create_transient(args)
            
            # Set host if provided
            if args.host_ra and args.host_dec:
                host_coords = SkyCoord(ra=args.host_ra*u.deg, 
                                     dec=args.host_dec*u.deg)
                transient.host = Host(
                    sky_coord=host_coords,
                    redshift=args.host_redshift
                )
            
            self.download_cutouts(transient, args)
            self.run_photometry(transient, args)
            
        elif args.command == 'sed':
            transient = self.create_transient(args)
            
            # Load photometry from file
            if args.photometry_file:
                self.load_photometry(transient, args.photometry_file)
            else:
                self.logger.error("Photometry file is required for SED fitting")
                sys.exit(1)
            
            self.run_sed_fitting(transient, args)
            
        elif args.command == 'cutouts':
            transient = self.create_transient(args)
            self.download_cutouts(transient, args)
            
        else:
            self.parser.print_help()
            sys.exit(1)

def main():
    """Main entry point"""
    cli = FrankenBlastCLI()
    cli.run()

if __name__ == '__main__':
    main()