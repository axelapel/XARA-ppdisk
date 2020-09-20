#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jan  6 13:49:58 2020

@author: Axel LAPEL
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal, optimize
import xaosim as xs
import xara
from xara.fitting import vertical_rim
import time
from astropy.io import fits as fts
import pdb


def vertical_InnerRim(isz=512, pscale=15, thick=250,
                      inc=0.85, PA=310, rad=800, cont=1/1000):
    """
    Parametric model that returns a square array (isz*isz) containing the image
    of the vertical rim of height thick seen with an inclination angle i of a
    transitional/proto-planetary disk of radius rad tilted by the position
    angle PA. A pixel star is also added.

    Parameters
    ----------
    isz    : int   // Image size (in pixels) corresponding to the properties of
                   // the CCD.
    pscale : float // Plate scale of the instrument (in mas/pixel).
    thick  : float // Thickness of the vertical inner rim (in mas).
    inc    : float // Inclination angle (in rad) between the observer and the
                   // disk. 0° : the disk appears as a circle.
    PA     : float // Position angle (in deg). Convention = trigo way
    rad    : float // Radius of the gap (in mas).
    cont   : float // Contrast defined as a ratio <1.
    """
    X, Y = np.meshgrid(np.arange(isz) - isz/2, np.arange(isz) - isz/2)
    eq1 = (X*pscale/rad)**2 + (Y*pscale/rad/np.cos(inc))**2
    eq2 = (X*pscale/rad)**2 + \
        ((Y-thick*np.sin(inc)/pscale)*pscale/rad/np.cos(inc))**2
    el1 = np.zeros((isz, isz))
    el2 = np.zeros((isz, isz))
    el1[eq1 <= 1.0] = 1.0
    el2[eq2 <= 1.0] = 1.0
    inside = el1 * el2                  # matching region of the ellipses
    crescent = (el2 - inside) * cont      # inner rim without rotation with PA
    crescent[int(isz/2), int(isz/2+thick*np.sin(inc))] = 1.0  # adding the star
    model = ndimage.rotate(crescent, -PA, reshape=False, order=0)
    return model


def imgcube_disk(isz, pscale, params, n_img):
    psf_cube = np.zeros((n_img, isz, isz))
    star_cube = np.copy(psf_cube)
    img_cube = np.copy(psf_cube)
    disk = vertical_rim(isz, pscale, params[0], params[1], params[2],
                        params[3], params[4])

    scexao = xs.instrument("SCExAO")
    scexao.atmo.update_screen(correc=10, rms=200)
    scexao.start()

    for i in range(n_img):
        psf = scexao.snap()
        psf = psf[:, 32:288]
        psf_cube[i] = psf
        time.sleep(0.2)

    for i in range(n_img):
        star = scexao.snap()
        star = star[:, 32:288]
        star_cube[i] = star
        time.sleep(0.2)

    scexao.stop()

    for i in range(n_img):
        img = signal.fftconvolve(disk, psf_cube[i], mode="same")
        img_cube[i] = img

    out = "data_disk.fits"
    hdu_out = fts.PrimaryHDU()
    hdu_out.data = img_cube
    hdu_out.header["KEYnew1"] = "Observation frames"
    hdu_out.writeto(out, overwrite=True)

    out = "data_star.fits"
    hdu_out = fts.PrimaryHDU()
    hdu_out.data = star_cube
    hdu_out.header["KEYnew1"] = "Calibration frames"
    hdu_out.writeto(out, overwrite=True)
    return img_cube, star_cube


def stat_calib_extraction(cube_disk, cube_calib, pscale, wl, kpo, kpo_cal):
    KPO.extract_KPD_single_cube(cube_disk, pscale, wl, recenter=True)
    KPO_cal.extract_KPD_single_cube(cube_calib, pscale, wl, recenter=True)
    data_disk = np.array(KPO.KPDT)[0]
    data_calib = np.array(KPO_cal.KPDT)[0]
    med_kphi_disk = np.median(data_disk, axis=0)
    med_kphi_calib = np.median(data_calib, axis=0)
    calibrated_stat_kerphi = med_kphi_disk - med_kphi_calib
    errors = np.sqrt(np.var(data_disk, axis=0)/(KPO.KPDT[0].shape[0] - 1))
    return calibrated_stat_kerphi, errors


def residuals(params, isz, pscale, KPO, wl, calib_kpdt, errors):
    model = vertical_rim(
        isz, pscale, params[0], params[1], params[2], params[3], params[4])
    kphi_th = grid_src_KPD(model, pscale, KPO.kpi, wl)
    res = (calib_kpdt - kphi_th) / errors
    # pdb.set_trace()
    return res


def image_simulation(model, PSF):
    """
    Returns an array of the cropped simulated image as the convolution of the
    object of interest with the given PSF.

    Parameters
    ----------
    model : (isz*isz) array containing the simulated obj. of given parameters.
    PSF   : (isz*isz) array containing the PSF of the telescope.
    """
    isz = np.shape(model)[0]
    image = signal.fftconvolve(model, PSF, mode="same")
    crop_img = image[isz//4:3*isz//4, isz//4:3*isz//4]  # avoid sampl. issues
    xx, yy = np.meshgrid(np.arange(isz/2)-isz/4, np.arange(isz/2)-isz/4)
    eq_circ = xx * xx + yy * yy
    mask = np.zeros((int(isz/2), int(isz/2)))
    mask[eq_circ <= (isz/4)**2] = 1.0
    crop_img = crop_img * mask
    return crop_img


def grid_src_KPD(mgrid, gscale, kpi, wl, phi=None, deg=False):
    cvis = xara.grid_src_cvis(kpi.uv[:, 0], kpi.uv[:, 1],
                              wl, mgrid, gscale, phi)
    kphase = np.dot(kpi.KerPhi, np.angle(cvis, deg=deg))
    return kphase


def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy > 0:
        X[:dy, :] = 0
    elif dy < 0:
        X[dy:, :] = 0
    if dx > 0:
        X[:, :dx] = 0
    elif dx < 0:
        X[:, dx:] = 0
    return X


###############################################################################
############### Parametric model for a vertical inner rim #####################
###############################################################################

if __name__ == "__main__":

    plt.rcParams["xtick.bottom"] = "True"
    plt.rcParams["ytick.left"] = "True"
    plt.rcParams["image.interpolation"] = "nearest"
    plt.rcParams["image.origin"] = "lower"
    plt.rcParams["figure.figsize"] = [6., 6.]

    darkblue_beamer = "#23373b"

    # ------------------------------ Scaling ----------------------------------
    isz = 256               # image size (in pixels)
    pscale = 16.7              # image plate scale (in mas/pixel)
    D = 7.92              # diameter of the telescope (in meters)
    ppscale = D / isz           # pupil pixel scale (in meters/pixel)
    rad2deg = 180 / np.pi       # convert radians to degrees
    deg2rad = np.pi / 180       # convert degrees to radians
    wl = 1.6e-6            # wavelength of the image  (in meters)
    m2pix = xara.core.mas2rad(pscale) * isz / wl  # convert meters to pixels

    # ----------------------------- Parameters --------------------------------
    # Note : set with LkCa15
    PA = 310                 # position angle (in deg)
    thick = 101              # true thickness of inner rim (in mas)
    inc = 60                 # inclination angle (in deg)
    rad = 450                # radius of the gap (in mas)
    cont = 1/10              # contrast (ratio <1)
    params = [thick, rad, cont, inc, PA]

    ############################### Modeling ##################################

    model = vertical_rim(isz, pscale, params[0], params[1],
                         params[2], params[3], params[4])

    ################# Image on the focus of Subaru (SCExAO) ###################

    # Creation of the pupil
    pupil_bool = xs.pupil.subaru(isz, isz, isz/2, True, True)
    pupil = pupil_bool.astype(int)
    scexao = xs.instrument("SCExAO")
    scexao.atmo.update_screen(correc=10, rms=200)
    PSF = scexao.snap()
    PSF = PSF[:, 32:288]

    # Convolution of the object with the PSF
    image = signal.fftconvolve(model, PSF, mode="same")
    crop_img = image[isz//4:3*isz//4, isz//4:3*isz//4]  # avoid sampling issues
    xx, yy = np.meshgrid(np.arange(isz/2) - isz/4, np.arange(isz/2) - isz/4)
    eq_circ = xx * xx + yy * yy
    mask = np.zeros((int(isz/2), int(isz/2)))
    mask[eq_circ <= (isz/4)**2] = 1.0
    crop_img = crop_img * mask

    # Calibration star
    xx, yy = np.meshgrid(np.arange(isz/2) - isz/4, np.arange(isz/2) - isz/4)
    eq_circ = xx * xx + yy * yy
    mask = np.zeros((int(isz/2), int(isz/2)))
    mask[eq_circ <= (isz/4)**2] = 1.0
    calib = PSF[isz//4:3*isz//4, isz//4:3*isz//4] * mask

    # Binary system
    star1 = PSF[isz//4:3*isz//4, isz//4:3*isz//4]
    star2 = shift_image(calib, 3, -2)
    contrast = 0.01
    binary = star1 + contrast * star2

    ############################ Kernel model #################################

    disc_pup = xara.core.create_discrete_model(pupil, ppscale, 0.3,
                                               binary=False, tmin=0.1)
    KPO = xara.KPO(array=disc_pup, bmax=7.92)
    KPO_cal = KPO.copy()
    KPO_bin = KPO.copy()
    KPO.extract_KPD_single_frame(crop_img, pscale, wl,
                                 target="Protoplanetary disk", recenter=False)
    KPO_cal.extract_KPD_single_frame(calib, pscale, wl,
                                     target="Calibration star", recenter=False)
    KPO_bin.extract_KPD_single_frame(binary, pscale, wl,
                                     target="Binary system", recenter=False)
    KPO.kpi.plot_pupil_and_uv(xymax=4.0, cmap="plasma_r",
                              ssize=9, figsize=(8, 4), marker='o')
    #U, S, Vt = np.linalg.svd(KPO.kpi.BLM)
    kerphi_disk = KPO.KPDT[0][0]
    kerphi_star = KPO_cal.KPDT[0][0]
    kerphi_bin = KPO_bin.KPDT[0][0]
    kerphi_calibrated = kerphi_disk - kerphi_star

    # Statistical processing for n_img frames
    n_img = 100
    cube_disk, cube_star = imgcube_disk(isz, pscale, params, n_img)
    KPO.extract_KPD_single_cube(cube_disk, pscale, wl, recenter=True)
    KPO_cal.extract_KPD_single_cube(cube_star, pscale, wl, recenter=True)
    data_disk = np.array(KPO.KPDT)[0]
    data_calib = np.array(KPO_cal.KPDT)[0]
    med_kphi_disk = np.median(data_disk, axis=0)
    med_kphi_calib = np.median(data_calib, axis=0)
    calibrated_stat_kerphi = med_kphi_disk - med_kphi_calib
    errors = np.sqrt((np.var(data_disk, axis=0) + np.var(data_calib, axis=0))
                     / (KPO.KPDT[0].shape[0] - 1))

    # Theoretical kernel phase
    kerphi_th = grid_src_KPD(model, pscale, KPO.kpi, wl)

    # Chi-2
    if errors is not None:
        chi2 = np.sum(((calibrated_stat_kerphi - kerphi_th) / errors)**2) \
            / KPO.kpi.nbkp
    else:
        chi2 = np.sum(((calibrated_stat_kerphi - kerphi_th))**2) / KPO.kpi.nbkp

    """# Parameters fitting
    p0 = np.array([100., 449., 0.11, 59., 311.])
    #params, cov = optimize.leastsq(
    #        residuals, p0, args=((calibrated_stat_kerphi, kerphi_th, errors)))
    params, cov = optimize.leastsq(
        residuals, p0, args=((isz, pscale, KPO,
                              wl, calibrated_stat_kerphi, errors)), factor=1)
    
    sol = optimize.least_squares(residuals, p0, args=(isz, pscale, KPO,
                              wl, calibrated_stat_kerphi, errors))"""

    ############################ Colinearity map ##############################

    gsize = int(isz/2)     # gsize x gsize grid
    gstep = int(pscale/2)  # grid step in mas
    xx, yy = np.meshgrid(
        np.arange(gsize) - gsize/2, np.arange(gsize) - gsize/2)

    col_map = KPO.kpd_binary_match_map(gsize, gstep, calibrated_stat_kerphi,
                                       cref=cont, norm=True)

    col_map_bin = KPO.kpd_binary_match_map(isz//5, pscale//5,
                                           kerphi_bin,
                                           cref=cont, norm=True)

    # ------------------------------ Plotting ---------------------------------

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    ax1.imshow(model**0.2, cmap="hot", label="Object")
    ax1.plot(int(isz/2), int(isz/2), marker="*",
             color="goldenrod", markersize=7)
    ax2.imshow(PSF**0.2, cmap="hot")
    ax3.imshow(image**0.4, cmap="hot")
    ax1.set_title("Object")
    ax2.set_title("SCExAO PSF")
    ax3.set_title("Image")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])

    """fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(9,3), sharey=True)
    ax4.plot(med_kphi_disk, color="lightblue")
    ax4.set_xlabel("Kernel-phase index")
    ax4.set_ylabel("Kernel-phase (rad)")
    ax4.set_title(r"$Ker-\phi$ : disk")
    ax5.plot(med_kphi_calib, color="darkturquoise")
    ax5.set_xlabel("Kernel-phase index")
    ax5.set_title(r"$Ker-\phi$ : calib. star")
    ax6.plot(calibrated_stat_kerphi, color="teal")
    ax6.set_xlabel("Kernel-phase index")
    ax6.set_title(r"$Ker-\phi$ : calibrated disk")"""

    fig2, ax6 = plt.subplots()
    ax6.plot(med_kphi_disk, color="tab:blue", label=r"$Ker-\phi$ : disk")
    ax6.plot(med_kphi_calib, color="goldenrod",
             label=r"$Ker-\phi$ : calib. star")
    ax6.set_xlabel("Kernel-phase index")
    ax6.set_ylabel("Kernel-phase (rad)")
    ax6.legend(loc="best")

    fig3, ax7 = plt.subplots()
    ax7.imshow(col_map, extent=(
        gsize/2*gstep, -gsize/2*gstep, -gsize/2*gstep, gsize/2*gstep))
    ax7.set_xlabel("Right ascension (mas)")
    ax7.set_ylabel("Declination (mas)")
    ax7.plot([0, 0], [0, 0], "*", color="goldenrod", ms=16)
    ax7.set_title("Calibrated signal colinearity map")
    ax7.grid()

    fig4, (ax8, ax9) = plt.subplots(1, 2, figsize=(10, 5))
    ax8.errorbar(kerphi_th, calibrated_stat_kerphi, yerr=errors,
                 fmt="none", ecolor="tab:blue", alpha=0.3)
    ax8.scatter(kerphi_th, calibrated_stat_kerphi, s=12,
                color="tab:blue")
    mmax = np.round(np.abs(calibrated_stat_kerphi).max())
    mmin = np.round(calibrated_stat_kerphi.min())
    ax8.plot([mmin, mmax+2], [mmin, mmax+2], "--", color="tab:red")
    ax8.set_ylabel("Data kernel-phase (rad)")
    ax8.set_xlabel("Model kernel-phase (rad)")
    ax8.set_title('Kernel-phase correlation diagram')
    ax8.grid()
    ax9.plot(calibrated_stat_kerphi, color="tab:blue",
             label="Calibrated data kernel phase")
    ax9.plot(kerphi_th, "--", color="tab:red",
             label="Theoretical kernel phase")
    ax9.legend(loc="best")
    ax9.set_xlabel("Kernel-phase index")
    ax9.set_ylabel("Kernel-phase (rad)")
    ax9.set_title("Superposition model/data")
    fig4.set_tight_layout(True)
