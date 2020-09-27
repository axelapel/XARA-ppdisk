import matplotlib.pyplot as plt
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

    path_model = "./models/vertical_rim.fits"
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
                                                      0.3,
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

    fig1, ax = plt.subplots()
    ax.plot(med_kphi_disk, color="tab:blue", label=r"$Ker-\phi$ : disk")
    ax.plot(med_kphi_calib, color="goldenrod",
            label=r"$Ker-\phi$ : calib. star")
    ax.set(xlabel="Kernel-phase index", ylabel="Kernel-phase (rad)")
    ax.legend(loc="best")
    fig1.savefig("plots/calibration.png", transparent=True)

    # Theoretical kernel phases
    kphi_th = grid_src_KPD(model, pscale, KPO_disk.kpi, wl)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.errorbar(kphi_th, calibrated_kphi, yerr=errors,
                 fmt="none", ecolor="tab:blue", alpha=0.3)
    ax1.scatter(kphi_th, calibrated_kphi, s=12,
                color="tab:blue")
    mmax = np.round(np.abs(calibrated_kphi).max())
    mmin = np.round(calibrated_kphi.min())
    ax1.plot([mmin, mmax+2], [mmin, mmax+2], "--", color="tab:red")
    ax1.set(xlabel="Model kernel-phase (rad)",
            ylabel="Data kernel-phase (rad)",
            title="Kernel-phase correlation diagram")
    ax1.grid()
    ax2.plot(calibrated_kphi, color="tab:blue",
             label="Calibrated data kernel phase")
    ax2.plot(kphi_th, "--", color="tab:red",
             label="Theoretical kernel phase")
    ax2.legend(loc="best")
    ax2.set(xlabel="Kernel-phase index", ylabel="Kernel-phase (rad)",
            title="Superposition model/data")
    fig2.set_tight_layout(True)

    # Chi2 estimation
    if errors is not None:
        chi2 = np.sum(((calibrated_kphi-kphi_th) / errors)
                      ** 2) / KPO_disk.kpi.nbkp
    else:
        chi2 = np.sum(((calibrated_kphi-kphi_th))**2) / KPO_disk.kpi.nbkp

    # Colinearity map
    # [Hypothesis that the disk is composed of many independent binaries]
    # gsize = int(isz/2)     # gsize x gsize grid
    gsize = int(1.5*isz)     # gsize x gsize grid
    gstep = int(pscale/2)  # grid step in mas
    xx, yy = np.meshgrid(np.arange(gsize) - gsize/2,
                         np.arange(gsize) - gsize/2)

    col_map = KPO_disk.kpd_binary_match_map(
        gsize, gstep, calibrated_kphi, cref=1e-1, norm=True)

    fig3, ax3 = plt.subplots()
    ax3.imshow(col_map, extent=(gsize/2*gstep, -gsize /
                                2*gstep, -gsize/2*gstep, gsize/2*gstep), cmap="viridis", origin="lower")
    ax3.set_xlabel("Right ascension (mas)")
    ax3.set_ylabel("Declination (mas)")
    ax3.plot([0, 0], [0, 0], "*", color="goldenrod", ms=16)
    ax3.set_title("Calibrated signal colinearity map")
    ax3.grid()
    fig3.savefig("plots/colinearity_map.png", transparent=True)
