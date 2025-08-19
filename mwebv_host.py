from dustmaps.sfd import SFDQuery
from classes import *

# Set up dustmaps with this
'''
# Use correct dustmap data directory
from dustmaps.config import config

config.reset()
config["data_dir"] = settings.DUSTMAPS_DATA_ROOT
'''

def get_dust_maps(position):
    """Gets milkyway reddening value"""

    ebv = SFDQuery()(position)
    # see Schlafly & Finkbeiner 2011 for the 0.86 correction term
    return 0.86 * ebv

def get_mwebv(transient):
	mwebv = get_dust_maps(transient.coordinates)
	return mwebv