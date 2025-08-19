import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import prospect.io.read_results as reader

#pretty plotting
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['axes.linewidth'] = 2.


def load_fit_data(hdf5_file, chain_file, modeldata_file, perc_file):
    """
    Load the observation data and best-fit model from HDF5 and NPZ files.
    """
    resultpars, obs, _ = reader.results_from(hdf5_file, dangerous=False)
    maggies = obs['maggies']
    maggies_unc = obs['maggies_unc']
    obs_filters = obs['filters']
    phot_obs_wave = obs['phot_wave']
    zred = obs['redshift']
    phot_rest_wave = np.array(phot_obs_wave) / (1 + zred)

    # Load the sampling chain
    best_fit_chain = np.load(chain_file, allow_pickle=True)["chains"]  # Use allow_pickle=True

    # Load the best-fit model data
    model_data = np.load(modeldata_file, allow_pickle=True)
    rest_wavelength = model_data["rest_wavelength"]
    spec = model_data["spec"]
    phot = model_data["phot"]
    spec_16 = model_data["spec_16"]
    spec_84 = model_data["spec_84"]

    # Load percentiles
    perc_data = np.load(perc_file, allow_pickle=True)["percentiles"]

    return {
        "maggies": maggies,
        "maggies_unc": maggies_unc,
        "filters": obs_filters,
        "best_fit_chain": best_fit_chain,
        "phot_rest_wave": phot_rest_wave,
        "rest_wavelength": rest_wavelength,
        "spec": spec,
        "phot": phot,
        "spec_16": spec_16,
        "spec_84": spec_84,
        "percentiles": perc_data,
    }



def plot_sed(data, transient_name, output_dir):
    """
    Plot the observed SED and best-fit model with wavelength in Angstroms.
    Parameters:
        data (dict): Dictionary containing observed and model data.
        transient_name (str): Name of the transient.
        output_dir (str): Directory to save the plot.
    """
    # Extract observed data
    obs_maggies = data["maggies"]
    obs_maggies_unc = data["maggies_unc"]
    obs_filters = data["filters"]

    # Attempt to extract best-fit model data
    best_phot = data.get("best_phot")
    best_spec = data.get("best_spec")
    phot_rest_wave = data.get("phot_rest_wave")
    rest_wavelength = data.get("rest_wavelength")
    phot_16 = data.get("phot_16")
    phot_84 = data.get("phot_84")
    spec_16 = data.get("spec_16")
    spec_84 = data.get("spec_84")
    spec = data.get("spec")
    phot = data.get("phot")


    # **Fix**: Convert rest_wavelength from nm to Angstroms if it's in nm (1 nm = 10 Angstroms)
    if np.max(rest_wavelength) < 1000:  # A simple check for likely nm values
        print("Rest wavelength seems to be in nanometers, converting to Angstroms...")
        rest_wavelength *= 10  # Convert nm to Angstroms

    # Debugging: Check again after conversion
    #print("Rest wavelength after conversion (Angstroms):", rest_wavelength[:10])

    # Convert maggies to flux for plotting
    flux_obs = obs_maggies * 3631e6  # mu-Jy
    flux_obs_unc = obs_maggies_unc * 3631e6  #mu-Jy

    plt.figure(figsize=(10, 6))

    # Plot observed data points with error bars using rest_wavelength in Angstroms
    plt.errorbar(
        phot_rest_wave,  
        flux_obs,
        yerr=flux_obs_unc,
        fmt="o",
        color="black",
        label="Observed Photometry",
    )

    # Plot best-fit photometry if available
    if best_phot is not None:
        flux_model = best_phot * 3631e6  # mu-Jy
        plt.scatter(
            phot_rest_wave, flux_model, color="indigo", label="Best-Fit Photometry"
        )

    # Plot best-fit spectrum if available
    if best_spec is not None and rest_wavelength is not None:
        flux_spec = best_spec * 3631e6  # mu-Jy
        plt.plot(
            rest_wavelength,
            flux_spec,
            color="darkorchid",
            label="Best-Fit Spectrum",
            linestyle="-",
        )

    # Add uncertainty ranges if available
    if phot_16 is not None and phot_84 is not None:
        plt.fill_between(
            phot_rest_wave,
            phot_16 * 3631e6,
            phot_84 * 3631e6,
            color="indigo",
            alpha=0.3,
            label="Photometry Uncertainty",
        )
    if spec_16 is not None and spec_84 is not None:
        plt.fill_between(
            rest_wavelength,
            spec_16 * 3631e6,
            spec_84 * 3631e6,
            color="darkorchid",
            alpha=0.3,
            label="Spectrum Uncertainty",
        )

    # Ensure at least the model legend is added, even if data is missing
    if best_phot is None:
        flux_model = phot * 3631e6  # mu-Jy
        plt.scatter(
            phot_rest_wave, flux_model, color="indigo", label="Model Photometry",
            marker = 's'
        )
    if best_spec is None:
        flux_spec = spec * 3631e6  # mu-Jy
        plt.plot(
            rest_wavelength,
            flux_spec,
            color="darkorchid",
            label="Model Spectrum",
            linestyle="-",
        )

    # Set the x-axis to logarithmic scale and label it in Angstroms
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Rest Wavelength (Angstroms)")
    plt.ylabel("Flux [$\mu$Jy]")
    #plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.title(f"{transient_name}")
    plt.xlim(1e3,3e5)
    plt.ylim(1e0, 1e8)
    plt.legend(loc='upper left')

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{transient_name}_sed_plot.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    # Read transient name from command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python plot_best_models.py <transient_name>")
        sys.exit(1)

    transient_name = sys.argv[1]
    base_dir = "./data/sed_output/"  # Define base directory for input/output

    # Define input and output paths
    hdf5_file = os.path.join(base_dir, transient_name, f"{transient_name}_global.h5")
    chain_file = os.path.join(base_dir, transient_name, f"{transient_name}_global_chain.npz")
    modeldata_file = os.path.join(base_dir, transient_name, f"{transient_name}_global_modeldata.npz")
    perc_file = os.path.join(base_dir, transient_name, f"{transient_name}_global_perc.npz")
    output_dir = os.path.join(base_dir, transient_name)

    # Check if required files exist
    for file in [hdf5_file, chain_file, modeldata_file, perc_file]:
        if not os.path.exists(file):
            print(f"File not found: {file}")
            sys.exit(1)

    # Load data
    data = load_fit_data(hdf5_file, chain_file, modeldata_file, perc_file)

    # Plot and save
    plot_sed(data, transient_name, output_dir)
