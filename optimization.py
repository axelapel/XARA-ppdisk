#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:09:16 2020

@author: alapel
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from astropy.io import fits as fts
import xaosim as xs
import xara
from xara.fitting import vertical_rim
import sys
import pdb


def cubefits_opening(fits):
    hdulist = fts.open(fits)
    hdu = hdulist[0]
    cube = hdu.data
    return cube


def stat_calib_extraction(cube_disk, cube_calib, pscale, wl, kpo, kpo_cal):
    kpo.extract_KPD_single_cube(cube_disk, pscale, wl, recenter=True)
    kpo_cal.extract_KPD_single_cube(cube_calib, pscale, wl, recenter=True)
    data_disk = np.array(kpo.KPDT)[0]
    data_calib = np.array(kpo_cal.KPDT)[0]
    med_kphi_disk = np.median(data_disk, axis=0)
    med_kphi_calib = np.median(data_calib, axis=0)
    calibrated_stat_kerphi = med_kphi_disk - med_kphi_calib
    errors = np.sqrt((np.var(data_disk, axis=0) + np.var(data_calib, axis=0))
                     / (KPO.KPDT[0].shape[0] - 1))
    return calibrated_stat_kerphi, errors


def grid_src_KPD(mgrid, gscale, kpi, wl, phi=None, deg=False):
    cvis = xara.grid_src_cvis(kpi.uv[:, 0], kpi.uv[:, 1],
                              wl, mgrid, gscale, phi)
    kphase = np.dot(kpi.KerPhi, np.angle(cvis, deg=deg))
    return kphase


def residuals(params, isz, pscale, KPO, wl, calib_kpdt, errors):
    model = vertical_rim(isz, pscale, params[0], params[1], params[2],
                         params[3], params[4])
    kphi_th = grid_src_KPD(model, pscale, KPO.kpi, wl)
    res = (calib_kpdt - kphi_th) / errors
    # pdb.set_trace()
    return res


if __name__ == "__main__":

    # General parameters
    isz = 256
    wl = 1.6e-6
    D = 7.92
    pscale = 16.7
    ppscale = D / isz

    # Model parameters
    PA = 310                 # position angle (in deg)
    h = 101                 # true thickness of inner rim (in mas)
    inc = 60                  # inclination angle (in deg)
    rad = 450                 # radius of the gap (in mas)
    cont = 1/10                # contrast (ratio <1)

    params = [h, rad, cont, inc, PA]

    # Extraction of the cubes
    calib_file = "data_star.fits"
    disk_file = "data_disk.fits"
    cube_calib = cubefits_opening(calib_file)
    cube_disk = cubefits_opening(disk_file)

    # Discrete model
    pupil_bool = xs.pupil.subaru(isz, isz, isz/2, True, True)
    pupil = pupil_bool.astype(int)
    disc_pup = xara.core.create_discrete_model(pupil, ppscale, 0.3,
                                               binary=False, tmin=0.1)
    KPO = xara.KPO(array=disc_pup, bmax=7.92)
    KPO_cal = KPO.copy()

    # Statistics on the cubes
    calibrated_stat_kerphi, errors = stat_calib_extraction(
        cube_disk, cube_calib, pscale, wl, KPO, KPO_cal)

    # Optimizing independantly the parameters

    """# Height (prior)
    hmin = 50
    hmax = 150
    hs   = np.arange(hmin, hmax)
    chi2_h = np.zeros(hmax-hmin)
    print("\nEvaluating the chi-2 for h :")
    
    for i in range(hmax-hmin):
        params[0] = hs[i]
        model     = vertical_rim(isz, pscale, params[0], params[1],
                                 params[2], params[3], params[4])
        kerphi_th = grid_src_KPD(model, pscale, KPO.kpi, wl)
        chi2_h[i] = np.sum(((calibrated_stat_kerphi - kerphi_th)/errors)**2)\
            / KPO.kpi.nbkp
        sys.stdout.write("\r%3d/%3d"%(i+1, hmax-hmin))
        sys.stdout.flush() 
    params = [h, rad, cont, inc, PA]
        
    # Radius (prior)
    rmin = 400
    rmax = 500
    rads = np.arange(rmin, rmax)
    chi2_rad = np.zeros(rmax-rmin)
    print("\nEvaluating the chi-2 for rad :")
    
    for i in range(rmax-rmin):
        params[1] = rads[i]
        model     = vertical_rim(isz, pscale, params[0], params[1],
                                 params[2], params[3], params[4])
        kerphi_th = grid_src_KPD(model, pscale, KPO.kpi, wl)
        chi2_rad[i] = np.sum(((calibrated_stat_kerphi - kerphi_th)/errors)**2)\
            / KPO.kpi.nbkp
        sys.stdout.write("\r%3d/%3d"%(i+1, rmax-rmin))
        sys.stdout.flush()
    params = [h, rad, cont, inc, PA]"""

    """# Contrast (prior)
    cmin = -20
    cmax = 20
    conts = np.arange(cmin, cmax)
    chi2_cont = np.zeros(cmax-cmin)
    print("\nEvaluating the chi-2 for cont :")
    
    for i in range(cmax-cmin):
        params[2] = conts[i]
        model     = vertical_rim(isz, pscale, params[0], params[1],
                                 params[2], params[3], params[4])
        kerphi_th = grid_src_KPD(model, pscale, KPO.kpi, wl)
        chi2_cont[i] = np.sum(((calibrated_stat_kerphi - kerphi_th)/errors)**2)\
            / KPO.kpi.nbkp
        sys.stdout.write("\r%3d/%3d"%(i+1, cmax-cmin))
        sys.stdout.flush()
    params = [h, rad, cont, inc, PA]"""

    # Contrast
    cmax = 0.000016
    nstep = 100
    step = cmax / nstep
    conts = np.arange(0, cmax, step)
    chi2_cont = np.zeros(nstep)
    print("\nEvaluating the chi-2 for cont :")

    for i in range(nstep):
        params[2] += i * step
        model = vertical_rim(isz, pscale, params[0], params[1],
                             params[2], params[3], params[4])
        kerphi_th = grid_src_KPD(model, pscale, KPO.kpi, wl)
        chi2_cont[i] = np.sum(((calibrated_stat_kerphi-kerphi_th)/errors)**2)\
            / KPO.kpi.nbkp
        sys.stdout.write("\r%3d/%3d" % (i+1, nstep))
        sys.stdout.flush()
    params = [h, rad, cont, inc, PA]

    """# Inclination (no prior)
    incmin = 0
    incmax = 90
    incs = np.arange(incmin, incmax)
    chi2_inc = np.zeros(incmax-incmin)
    print("\nEvaluating the chi-2 for inc :")
    
    for i in range(incmax-incmin):
        params[3] = incs[i]
        model     = vertical_rim(isz, pscale, params[0], params[1],
                                 params[2], params[3], params[4])
        kerphi_th = grid_src_KPD(model, pscale, KPO.kpi, wl)
        chi2_inc[i] = np.sum(((calibrated_stat_kerphi - kerphi_th)/errors)**2)\
            / KPO.kpi.nbkp
        sys.stdout.write("\r%3d/%3d"%(i+1, incmax-incmin))
        sys.stdout.flush()
    params = [h, rad, cont, inc, PA]
    
    # PA (prior)
    PAmin = 270
    PAmax = 360
    PAs   = np.arange(PAmin, PAmax)
    chi2_PA = np.zeros(PAmax-PAmin)
    print("\nEvaluating the chi-2 for PA :")
    
    for i in range(PAmax-PAmin):
        params[4] = PAs[i]
        model     = vertical_rim(isz, pscale, params[0], params[1],
                                 params[2], params[3], params[4])
        kerphi_th = grid_src_KPD(model, pscale, KPO.kpi, wl)
        chi2_PA[i] = np.sum(((calibrated_stat_kerphi - kerphi_th)/errors)**2)\
            / KPO.kpi.nbkp
        sys.stdout.write("\r%3d/%3d"%(i+1, PAmax-PAmin))
        sys.stdout.flush()
    params = [h, rad, cont, inc, PA]"""

    # Least-square optimization
    """p0 = np.array([100., 449., 0.11, 59., 311.])
    print("\n\nSeeking for the best parameters...\n")
    results, cov = optimize.leastsq(
        residuals, p0, args=((isz, pscale, KPO,
                              wl, calibrated_stat_kerphi, errors)), factor=1.)
    print("h = {}".format(results[0]))
    print("rad = {}".format(results[1]))
    print("cont = {}".format(results[2]))
    print("inc = {}".format(results[3]))
    print("PA = {}".format(results[4]))"""

    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    ax1.plot(hs, chi2_h, color="teal", label="h = 101.00 mas")
    ax1.grid()
    ax1.set_ylabel("Reduced " + r"$\chi^2$")
    ax1.set_xlabel("Height (mas)")
    ax1.set_xlim(50, 149)
    ax2.plot(rads, chi2_rad, color="goldenrod", label="r = 450.00 mas")
    ax2.grid()
    ax2.set_xlim(400, 499)
    ax2.set_ylabel("Reduced " + r"$\chi^2$")
    ax2.set_xlabel("Radius (mas)")
    ax3.plot(conts, chi2_cont, color="tab:red", label="c = 0.83e-5")
    ax3.grid()
    ax3.set_ylabel("Reduced " + r"$\chi^2$")
    ax3.set_xlabel("Contrast")
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax3.set_xlim(0, 1.55e-5)
    ax4.plot(incs, chi2_inc, color="tab:green", label="i = 60.00 deg")
    ax4.grid()
    ax4.set_xlim(1, 89)
    ax4.set_ylabel("Reduced " + r"$\chi^2$")
    ax4.set_xlabel("Inclination (deg)")
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    ax3.legend(loc="best")
    ax4.legend(loc="best")
    fig1.subplots_adjust(wspace=0.25, hspace=0.25)
    fig.savefig("4optimization.pdf")

    fig2, ax5 = plt.subplots()
    ax5.plot(PAs, chi2_PA, color="tab:purple", label="PA = 310.00 mas")
    ax5.grid()
    ax5.set_ylabel("Reduced " + r"$\chi^2$")
    ax5.set_xlabel("Position angle (deg)")
    ax5.set_xlim(276, 358)
    ax5.set_ylim(0)
    ax5.legend(loc="best")
    fig2.savefig("last_optimization.pdf")
