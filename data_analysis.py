import numpy as np
import xaosim as xs
import xara
from astropy.io import fits


def grid_src_KPD(mgrid, gscale, kpi, wl, phi=None, deg=False):
    """Generate kernel phases data from a model.

    Parameters
    ----------
    mgrid : array
        Model.
    gscale : float
        Plate scale.
    kpi : xara.KPI
        Kernel Phase Information object.
    wl : float
        Wavelength.
    phi : array, optional
        Pre-computed auxilliary array to speed up calculation, by default None
    deg : bool, optional
        Return angle in degrees if True, by default False

    Returns
    -------
    Array containing kernel phases.
    """
    cvis = xara.grid_src_cvis(kpi.uv[:, 0], kpi.uv[:, 1],
                              wl, mgrid, gscale, phi)
    kphase = np.dot(kpi.KerPhi, np.angle(cvis, deg=deg))
    return kphase


if __name__ == "__main__":

    # Files
    path_simu = "./fits_simu/"
    disk_cube = fits.getdata(path_simu + "disk_imgs.fits")
    calib_cube = fits.getdata(path_simu + "calib_imgs.fits")

    path_model = "./fits_models/vertical_rim.fits"
    model = fits.getdata(path_model)
    # Parameters
    isz = disk_cube.shape[0]
    pscale = 16.7              # image plate scale (in mas/pixel)
    D = 7.92                   # diameter of the telescope (in meters)
    ppscale = D / isz          # pupil pixel scale (in meters/pixel)
    wl = 1.6e-6                # wavelength of the image : H band (in meters)

    # Conversion
    rad2deg = 180. / np.pi      # convert radians to degrees
    deg2rad = np.pi / 180.      # convert degrees to radians

    # Creation of the pupil
    pupil_bool = xs.pupil.subaru(isz, isz, isz/2, True, True)
    pupil = pupil_bool.astype(int)
    discretized_pup = xara.core.create_discrete_model(pupil,
                                                      ppscale,
                                                      0.16,
                                                      binary=False,
                                                      tmin=0.1)

    # KPOs
    KPO_disk = xara.KPO(array=discretized_pup, bmax=D)
    KPO_calib = KPO_disk.copy()

    KPO_disk.extract_KPD_single_cube(disk_cube, pscale, wl,
                                     target="ppdisk", recenter=True)
    KPO_calib.extract_KPD_single_cube(calib_cube, pscale, wl,
                                      target="calibration_star", recenter=True)

    # Extraction of kernel phases
    data_disk = np.array(KPO_disk.KPDT)[0]
    data_calib = np.array(KPO_calib.KPDT)[0]

    # Calibration
    med_kphi_disk = np.median(data_disk, axis=0)
    med_kphi_calib = np.median(data_calib, axis=0)
    calibrated_kphi = med_kphi_disk - med_kphi_calib
    errors = np.sqrt((np.var(data_disk, axis=0) + np.var(data_calib, axis=0))
                     / (KPO_disk.KPDT[0].shape[0] - 1))

    # Theoretical kernal phase
    kphi_th = grid_src_KPD(model, pscale, KPO_disk.kpi, wl)

    # Chi2 estimation
    if errors is not None:
        chi2 = np.sum(((calibrated_kphi-kphi_th) / errors)
                      ** 2) / KPO_disk.kpi.nbkp
    else:
        chi2 = np.sum(((calibrated_kphi-kphi_th))**2) / KPO_disk.kpi.nbkp
