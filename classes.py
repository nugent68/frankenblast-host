from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import os

class Survey:
    """
    Class to represent a survey.
    """

    # Class-level list to store all Survey instances
    _surveys = []

    def __init__(self, name: str):
        self.name = name

        # Add this instance to the global list
        Survey._surveys.append(self)

    @classmethod
    def all(cls):
        """
        Return all Survey instances.
        """
        return cls._surveys

    def __str__(self):
        return self.name



class Filter:
    """
    Class to represent a survey filter.
    """
    # Class-level list to store all Filter instances
    _filters = []

    def __init__(
        self,
        name: str,
        survey: str,
        sedpy_name: str = None,
        sedpy_id: str = None,
        hips_id: str = None,
        vosa_id: str = None,
        image_download_method: str = None,
        pixel_size_arcsec: float = None,
        image_fwhm_arcsec: float = None,
        wavelength_eff_angstrom: float = None,
        wavelength_min_angstrom: float = None,
        wavelength_max_angstrom: float = None,
        vega_zero_point_jansky: float = None,
        magnitude_zero_point: float = None,
        ab_offset: float = None,
        magnitude_zero_point_keyword: str = None,
        image_pixel_units: str = None,
    ):
        self.name = name
        self.survey = survey
        self.sedpy_name = sedpy_name
        self.sedpy_id = sedpy_id
        self.hips_id = hips_id
        self.vosa_id = vosa_id
        self.image_download_method = image_download_method
        self.pixel_size_arcsec = pixel_size_arcsec
        self.image_fwhm_arcsec = image_fwhm_arcsec
        self.wavelength_eff_angstrom = wavelength_eff_angstrom
        self.wavelength_min_angstrom = wavelength_min_angstrom
        self.wavelength_max_angstrom = wavelength_max_angstrom
        self.vega_zero_point_jansky = vega_zero_point_jansky
        self.magnitude_zero_point = magnitude_zero_point
        self.ab_offset = ab_offset
        self.magnitude_zero_point_keyword = magnitude_zero_point_keyword
        self.image_pixel_units = image_pixel_units

        # Add this instance to the global list
        Filter._filters.append(self)

    def __str__(self):
        return self.name

    @classmethod
    def all(cls):
        """
        Return all Filter instances.
        """
        return cls._filters

    @property
    def transmission_curve(self):
        """
        Returns the transmission curve of the filter.

        Parameters
        ----------
        transmission_curves_root : str
            Path to the root directory containing transmission curve files.
        
        Returns
        -------
        observate.Filter (or other object depending on implementation)
        """
        transmission_curve_root = './data/transmission/'
        curve_name = f"{transmission_curve_root}{self.name}.txt"

        try:
            transmission_curve = pd.read_csv(curve_name, sep=r"\s+", header=None)
        except Exception as err:
            raise ValueError(
                f"{self.name}: Problem loading filter transmission curve from {curve_name}: {err}"
            )

        wavelength = transmission_curve[0].to_numpy()
        transmission = transmission_curve[1].to_numpy()
        # Replace `observate.Filter` with an appropriate object or processing logic
        return {"wavelength": wavelength, "transmission": transmission}

    def correlation_model(self):
        """
        Returns the model for correlated errors of the filter, if it exists.

        Parameters
        ----------
        transmission_curves_root : str
            Path to the root directory containing correlation model files.
        
        Returns
        -------
        tuple (array, array) or (None, None)
        """
        transmission_curves_root = './data/transmission/'

        corr_model_name = (
            f"{transmission_curves_root}/{self.name}_corrmodel.txt"
        )
        if not os.path.exists(corr_model_name):
            return None, None

        try:
            corr_model = pd.read_csv(corr_model_name, sep=r"\s+", header=None)
        except Exception as err:
            raise ValueError(
                f"{self.name}: Problem loading filter correlation model from {corr_model_name}: {err}"
            )

        app_radius = corr_model[0].to_numpy()
        error_adjust = corr_model[1].to_numpy() ** 0.5
        return app_radius, error_adjust




class Transient:
    """
    Model to represent a transient.
    """

    def __init__(self, name: str, coordinates: SkyCoord,
                 tns_id: str = None, milkyway_dust_reddening: float = None, transient_redshift: float = None):
        """
        Initialize a Transient instance.

        Parameters
        ----------
        name : str
            Name of the transient.
        coordinates : SkyCoord
            Sky coordinates of the transient.
        tns_id : str, optional
            TNS ID of the transient.
        milkyway_dust_reddening : float, optional
            Milky Way dust reddening value.
        transient_redshift : float, optional
            Spectroscopic redshift of transient
        """
        self.name = name
        self.coordinates = coordinates
        self.tns_id = tns_id
        self.milkyway_dust_reddening = milkyway_dust_reddening
        self.transient_redshift = transient_redshift
        self._host = None  # Private attribute to store the host
        self.global_apertures = None
        self.host_photometry = None
        self.host_phot_filters = None

    @property
    def host(self):
        """
        Host property to get the associated Host object.

        Returns
        -------
        Host
            The host galaxy associated with this transient.
        """
        return self._host

    @host.setter
    def host(self, host_obj):
        """
        Host property setter to associate a Host object with this transient.

        Parameters
        ----------
        host_obj : Host
            A Host object to associate with the transient.
        """
        if not isinstance(host_obj, Host) and host_obj is not None:
            raise TypeError("host must be an instance of Host or None")
        self._host = host_obj

    @property
    def cutouts(self):
        """
        Dynamically fetch all available cutouts for this transient.

        Returns
        -------
        list
            List of dictionaries, each containing the file path and associated Filter object.
        """
        base_dir = "./cutouts/"
        cutouts_dir = os.path.join(base_dir, self.name)
        if not os.path.exists(cutouts_dir):
            return []

        filters = Filter.all()  # Get all filter instances
        available_cutouts = []

        for root, _, files in os.walk(cutouts_dir):
            for file in files:
                if file.endswith(".fits"):
                    file_path = os.path.join(root, file)
                    filter_name = os.path.splitext(file)[0]  # Remove ".fits"
                    filter_obj = next((f for f in filters if f.name == filter_name), None)
                    if filter_obj:
                        available_cutouts.append({"file_path": file_path, "filter": filter_obj})

        return available_cutouts

    def __str__(self):
        """
        String representation of the Transient instance.
        """
        return f"Transient(name={self.name}, coordinates={self.coordinates}, tns_id={self.tns_id}, redshift={self.transient_redshift}, host={self.host})"
class Host:
    """
    Class to represent a host galaxy.
    """

    def __init__(self, name=None, redshift=None, 
        photometric_redshift=None, 
        milkyway_dust_reddening=None,
        coordinates=None, host_prob = None, smallcone_prob=None, missedcat_prob=None, association_catalog=None):
        self.name = name
        self.redshift = redshift
        self.photometric_redshift = photometric_redshift
        self.milkyway_dust_reddening = milkyway_dust_reddening
        self.coordinates = coordinates
        self.host_prob = host_prob
        self.missedcat_prob = missedcat_prob
        self.smallcone_prob = smallcone_prob
        self.association_catalog = association_catalog

    def __str__(self):
        return f"Host(name={self.name}, redshift={self.redshift}, photometric_redshift={self.photometric_redshift}, milkyway_dust_reddening={self.milkyway_dust_reddening}, host_prob={self.host_prob}, missedcat_prob{self.missedcat_prob}, smallcone_prob={self.smallcone_pron}, association_catalog={self.association_catalog})"

