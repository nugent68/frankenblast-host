# frankenblast-host
Code for rapid transient host association, photometry, and stellar population modeling

If using FrankenBlast, please cite Nugent et al. 2025. FrankenBlast is a customized version of the Blast web application (https://blast.scimma.org/), so we suggest you also cite Jones et al. 2024 (https://ui.adsabs.harvard.edu/abs/2024arXiv241017322J/abstract).

For installation: 
  - requirements.txt has the exact conda environment setup needed for FrankenBlast
  - settings.sh has some CUSTOM path files you need to put in your .zshrc or .bashrc or .bash_profile file
  - FrankenBlast currently uses Python 3.10.15
  - Make sure you pip or conda install: numpy, scipy, astro-prospector, python-fsps (and FSPS), sedpy, astro-prost, astroquery, corner, dustmaps, extinction, jupyter, pandas, sbi, seaborn, tensorflow, and matplotlib
 

The SBI++ trained models for the host SED fit are too large for GitHub, please go to [INSERT ZENODO LINK, TEMPORARY FILE ACCESS: https://drive.google.com/drive/folders/1AxmboHdbNvwAtNvTuHK0C38vPQ0SKQQN?usp=sharing] to download sbipp_phot.zip and sbi_training_sets.zip. Move these files to the ./data/ directory and unzip. 

FrankenBlast Tutorial.ipynb contains a tutorial for how to use FrankenBlast. We have also included files to do SBI++ host SED fitting with LSST data. A tutorial will be added soon.

CURRENT ISSUES:

We are currently experiencing an issue with the updated astro_prost PanSTARRS host association. To bypass this error, please edit the helpers.py file in astro_prost and comment out lines 2829-2830 (line 2829: galaxies["redshift_mean"] = np.nan, line 2830: galaxies["redshift_std"] = np.nan). This will return "0" as the redshift of the host, so please be aware of that! The other option is to not run astro_prost with PanSTARRS: to do this, you can simply delete "panstarrs" from the "catalogs" dictionary in line 278 in FrankenBlast's get_host.py.

Do not hesitate to reach out to Anya Nugent (anya.nugent[at]cfa.harvard.edu) for any questions or comments.
