# frankenblast-host
Code for rapid transient host association, photometry, and stellar population modeling

If using FrankenBlast, please cite Nugent et al. 2025. FrankenBlast is a customized version of the Blast web application (https://blast.scimma.org/), so we suggest you also cite Jones et al. 2024 (https://ui.adsabs.harvard.edu/abs/2024arXiv241017322J/abstract).

For installation: 
  - requirements.txt has the exact conda environment setup needed for FrankenBlast
  - settings.sh has some CUSTOM path files you need to put in your .zshrc or .bashrc or .bash_profile file
  - FrankenBlast currently uses Python 3.10.15
  - Make sure you install: numpy, scipy, astro-prospector, python-fsps (and FSPS), sedpy, astro-prost, astroquery, corner, dustmaps, extinction, jupyter, pandas, sbi, seaborn, tensorflow, and matplotlib

CURRENT ISSUES:

We are currently experiencing an issue with the updated astro_prost PanSTARRS host association. To bypass this error, please edit the helpers.py file in astro_prost and comment out lines 2829-2830 (line 2829: galaxies["redshift_mean"] = np.nan, line 2830: galaxies["redshift_std"] = np.nan). This will return "0" as the redshift of the host, so please be aware of that! The other option is to not run astro_prost with PanSTARRS: to do this, you can simply delete "panstarrs" from the "catalogs" dictionary in line 278 in FrankenBlast's get_host.py.

Do not hesitate to reach out to Anya Nugent (anya.nugent[at]cfa.harvard.edu) for any questions or comments.
