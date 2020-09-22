#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def vertical_rim(gsz, gstep, height, rad, cont, inc, PA):
    """ Parametric model of the inner edge of a circumstellar disk.
    ------------------------------------------------------------------
    Returns the square (gsz x gsz) grid size model of the vertical rim
    of a circumstellar disk of radius *rad*, height *thick* seen at
    inclination *inc* and for the position angle *PA* surrounding a
    bright star (luminosity contrast disk/contrast *cont*).

    Parameters:
    ----------
    - gsz    : grid size in pixels (int)
    - gstep  : grid step size in mas (float)
    - height : vertical inner rim height in mas (float)
    - rad    : radius of the gap (in mas). 
    - cont   : disk/star contrast ratio (0 < cont < 1)
    - inc    : inclination of the disk (in deg)
    - PA     : disk position angle E. of N. (in deg)
    ------------------------------------------------------------------ """

    xx, yy = np.meshgrid(gstep * (np.arange(gsz) - gsz/2),
                         gstep * (np.arange(gsz) - gsz/2))

    PA1 = PA * np.pi / 180    # convert position angle to radians
    inc1 = inc * np.pi / 180  # convert inclination to radians

    xx1 = xx * np.cos(PA1) + yy * np.sin(PA1)
    yy1 = yy * np.cos(PA1) - xx * np.sin(PA1)

    happ_thick = height * np.sin(inc1) / 2  # half apparent thickness

    dist1 = np.hypot((yy1 - happ_thick) / (rad * np.cos(inc1)), xx1 / rad)
    dist2 = np.hypot((yy1 + happ_thick) / (rad * np.cos(inc1)), xx1 / rad)

    el1 = np.zeros_like(dist1)
    el2 = np.zeros_like(dist2)

    el1[dist1 < 1] = 1.0
    el2[dist2 < 1] = 1.0

    rim = (el1 * (1 - el2))    # inner rim before rotation
    rim *= cont / rim.sum()
    rim[gsz//2, gsz//2] = 1.0  # adding the star
    return rim

    # TODO : correct the terms to appropriate one for disks (ex: inner edge/rim
    # is actually the first hole's edge)

    # TODO : precise that this is a model for one pit (hole) and verify in which
    # band it is legitimate


if __name__ == "__main__":

    # Imshow preferences
    plt.rcParams["xtick.bottom"] = "False"
    plt.rcParams["ytick.left"] = "False"
    plt.rcParams["xtick.labelbottom"] = "False"
    plt.rcParams["ytick.labelleft"] = "False"
    plt.rcParams["image.origin"] = "lower"
    plt.rcParams["image.cmap"] = "hot"

    # ------------------------------ Scaling ----------------------------------
    gsz = 256                  # grid size (in pixels)
    gstep = 15                 # grid step size (in mas)
    """pscale = 16.7              # image plate scale (in mas/pixel)
    D = 7.92                   # diameter of the telescope (in meters)
    ppscale = D / isz          # pupil pixel scale (in meters/pixel)
    rad2deg = 180 / np.pi      # convert radians to degrees
    deg2rad = np.pi / 180      # convert degrees to radians
    wl = 1.6e-6                # wavelength of the image : H band (in meters)
    m2pix = xara.core.mas2rad(pscale) * isz / wl  # convert meters to pixels"""

    # TODO : verify that the H band is relevant for this kind of view of disks
    # TODO : check the caracteristics of subaru/SCEXAO (if it works at the wavelengths
    # where we can see a banana and base everything on it if yes

    # ----------------------------- Parameters --------------------------------
    # Note : set with LkCa15
    PA = 310                 # position angle (in deg)
    height = 100             # true height of inner rim (in mas)
    inc = 60                 # inclination angle (in deg)
    rad = 450                # radius of the gap (in mas)
    cont = 1e-2              # contrast (ratio <1)

    # TODO : check the actual relevant parameters to evaluate from Radmc3d
    # TODO : check if there would be a better disk to observe from the METEOR with
    # Alexis at the relevant wavelength for Subaru and if so base the parameters on it

    # ----------------------------- Simulation --------------------------------
    model = vertical_rim(gsz, gstep, height, rad, cont, inc, PA)

    out = "./models/vertical_rim.fits"
    hdu = fits.PrimaryHDU(model)
    hdu.writeto(out, overwrite=True)

    # Display
    fig, ax = plt.subplots()
    ax.imshow(model**0.3, cmap="hot")
