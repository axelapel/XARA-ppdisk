#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import xaosim as xs
from astropy.io import fits
from scipy import signal

# Aliases
fft = np.fft.fft2
ifft = np.fft.ifft2
shift = np.fft.fftshift


def wfgen(dim, L, L0, wl, r0):
    """
    Wavefront generator using Von Karman model.

    Returns
    -------
    Both real and imaginary part of the wavefront as arrays of size dim*dim in
    an (1,2) array screen.
    """
    # (dim*dim) array of occurences of a random variable following a uniform
    # distribution between -pi and pi
    phase = np.random.uniform(-np.pi, np.pi, (dim, dim))
    x = np.arange(-dim/2, dim/2, 1)
    y = np.arange(-dim/2, dim/2, 1)
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X, Y)
    modulus = (r**2+(L/L0)**2)**(-11./12.)  # Von Karman model
    screen = shift(ifft(shift(modulus*np.exp(np.array([0.+1.j])*phase))))\
        * dim*dim
    screen *= np.sqrt(2.)*np.sqrt(0.0228)*(L/r0)**(5./6.)*wl/(2.*np.pi)
    screen -= np.mean(screen)
    return screen


def image(pupil, wl, wf=None, turb=False):
    """Convolution function with or without turbulence and a given pupil.

    Returns
    -------
    Simulated image.
    """
    if turb == True:
        img = pupil*np.exp(1j*2.*np.pi/wl*wf.real*pupil)
        img = shift(fft(shift(img)))
        img = np.abs(img)**2
    else:
        img = np.abs(shift(fft(shift(pupil))))**2
    return img


if __name__ == "__main__":

    # Imshow preferences
    plt.rcParams["xtick.bottom"] = "False"
    plt.rcParams["ytick.left"] = "False"
    plt.rcParams["xtick.labelbottom"] = "False"
    plt.rcParams["ytick.labelleft"] = "False"
    plt.rcParams["image.origin"] = "lower"
    plt.rcParams["image.cmap"] = "hot"

    ############################### Parameters ################################

    # Turbulence
    diam = 7.92
    L = 3 * diam     # L > 2 * diam (Nyquist)
    L0 = 20.
    r0 = .22         # Seen in literature
    wl = 1.6e-6

    ############################ Model extraction #############################

    file = "./models/vertical_rim.fits"
    model = fits.getdata(file)
    gsz = np.shape(model)[0]

    ################# Images on the focus of Subaru (SCExAO) ##################

    """
    # Creation of the pupil
    pupil_bool = xs.pupil.subaru(gsz, gsz, gsz/2, True, True)
    pupil = pupil_bool.astype(int)
    
    # Adding turbulence to the signal
    screen = wfgen(gsz, L, L0, wl, r0)
    model *= np.real(screen)
    
    # Image
    img = image(pupil, wl, model, True)
    """

    # PSF with turbulence
    scexao = xs.instrument("SCExAO")
    scexao.atmo.update_screen(correc=10, rms=250)
    PSF = scexao.snap()
    scexao.start()
    dim_1, dim_2 = np.shape(PSF)

    # PSF cube with turbulence
    N = 100  # Number of frames
    PSF_cube = np.zeros((N, dim_1, dim_2))
    for i in range(N):
        scexao.atmo.update_screen(correc=10, rms=250)
        PSF_cube[i] = scexao.snap()
    scexao.stop()

    # --------------------------------- Images ---------------------------------

    # pp-disk
    disk_imgs = np.zeros((N, gsz, gsz))
    for i in range(N):
        disk_imgs[i] = signal.fftconvolve(model, PSF_cube[i], mode="same")

    # Calibration star
    star_imgs = np.zeros_like(disk_imgs)
    for i in range(N):
        star_imgs[i] = PSF_cube[i, :, 32:288]

    # Saving data
    out1 = "./fits_simu/disk_imgs.fits"
    out2 = "./fits_simu/calib_imgs.fits"
    hdu_disk = fits.PrimaryHDU(disk_imgs)
    hdu_star = fits.PrimaryHDU(star_imgs)
    hdu_disk.writeto(out1, overwrite=True)
    hdu_star.writeto(out2, overwrite=True)

    # Convolution relation
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    ax1.imshow(model**0.3, cmap="hot", label="Object")
    ax1.plot(int(gsz/2), int(gsz/2), marker="*",
             color="goldenrod", markersize=7)
    ax2.imshow(PSF**0.3, cmap="hot")
    ax3.imshow(disk_imgs[0]**0.3, cmap="hot")
    ax1.set_title("Object")
    ax2.set_title("SCExAO PSF")
    ax3.set_title("Image")
    fig.savefig("plots/convolution.png", transparent=True)
