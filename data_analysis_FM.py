#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pf
import xaosim as xs
import xara
import sys
import pdb&

tgt_cube = pf.getdata('tgt_cube.fits')  # alpha Ophiuchi
# tgt_cube = pf.getdata('binary_test.fits') # fake binary added to eps Her
ca2_cube = pf.getdata('ca2_cube.fits')  # epsilon Herculis

pscale = 25.0              # plate scale of the image in mas/pixels
wl = 2.145e-6          # central wavelength in meters (Hayward paper)
ISZ = tgt_cube.shape[1]  # image size
kpo1 = xara.KPO(fname="p3k_med_grey_model.fits")
kpo2 = kpo1.copy()

kpo1.extract_KPD_single_cube(
    tgt_cube, pscale, wl, target="alpha Ophiuchi", recenter=True)
kpo2.extract_KPD_single_cube(
    ca2_cube, pscale, wl, target="epsilon Herculis", recenter=True)

data1 = np.array(kpo1.KPDT)[0]
data2 = np.array(kpo2.KPDT)[0]

mydata = np.median(data1, axis=0) - np.median(data2, axis=0)
myerr = np.sqrt(np.var(data1, axis=0) / (kpo1.KPDT[0].shape[0] - 1) + np.var(
    data2, axis=0) / (kpo2.KPDT[0].shape[0] - 1))
myerr = np.sqrt(myerr**2 + 1.2**2)

pdb.set_trace()

# ==========================================================
# median colinearity map
# ==========================================================
print("\ncomputing colinearity map...")
gsize = 100  # gsize x gsize grid
gstep = 10  # grid step in mas
xx, yy = np.meshgrid(
    np.arange(gsize) - gsize/2, np.arange(gsize) - gsize/2)
azim = -np.arctan2(xx, yy) * 180.0 / np.pi
dist = np.hypot(xx, yy) * gstep

#mmap = kpo1.kpd_binary_match_map(100, 10, mydata/myerr, norm=True)
mmap = kpo1.kpd_binary_match_map(100, 10, mydata, norm=True)
x0, y0 = np.argmax(mmap) % gsize, np.argmax(mmap) // gsize
print("max colinearity found for sep = %.2f mas and ang = %.2f deg" % (
    dist[y0, x0], azim[y0, x0]))

f1 = plt.figure(figsize=(5, 5))
ax1 = f1.add_subplot(111)
ax1.imshow(mmap, extent=(
    gsize/2*gstep, -gsize/2*gstep, -gsize/2*gstep, gsize/2*gstep))
ax1.set_xlabel("right ascension (mas)")
ax1.set_ylabel("declination (mas)")
ax1.plot([0, 0], [0, 0], "w*", ms=16)
ax1.set_title("Calibrated signal colinearity map")
ax1.grid()
f1.set_tight_layout(True)
f1.canvas.draw()
15
pdb.set_trace()

# ==========================================================
# model fitting
# ==========================================================
print("\nbinary model fitting...")
p0 = [dist[y0, x0], azim[y0, x0], mmap.max()]  # good starting point

mfit = kpo1.binary_model_fit(p0, calib=kpo2)
p1 = mfit[0]
# ==========================================================
#         correlation plot for best fit
# ==========================================================
cvis_b = xara.core.cvis_binary(
    kpo1.kpi.UVC[:, 0], kpo1.kpi.UVC[:, 1], wl, p1)  # binary
ker_theo = kpo1.kpi.KPM.dot(np.angle(cvis_b))

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

ax.errorbar(ker_theo, mydata, yerr=myerr, fmt="none", ecolor='c')
ax.plot(ker_theo, mydata, 'b.')
mmax = np.round(np.abs(mydata).max())
ax.plot([-mmax, mmax], [-mmax, mmax], 'r')
ax.set_ylabel("data kernel-phase")
ax.set_xlabel("model kernel-phase")
ax.set_title('kernel-phase correlation diagram')
ax.axis("equal")
ax.axis([-11, 11, -11, 11])
ax.grid()
fig.set_tight_layout(True)

if myerr is not None:
    chi2 = np.sum(((mydata - ker_theo)/myerr)**2) / kpo1.kpi.nbkp
else:
    chi2 = np.sum(((mydata - ker_theo))**2) / kpo1.kpi.nbkp

print("sep = %3f, ang=%3f, con=%3f => chi2 = %.3f" %
      (p1[0], p1[1], p1[2], chi2))
print("correlation matrix of parameters")
print(np.round(mfit[1], 2))

# ==============================================================
# other tests to check for chi2 values
# ==============================================================
p2 = [124.5, 86.5, 25.0]
cvis_b = xara.core.cvis_binary(
    kpo1.kpi.UVC[:, 0], kpo1.kpi.UVC[:, 1], wl, p2)  # for alpha oph
ker_theo = kpo1.kpi.KPM.dot(np.angle(cvis_b))

if myerr is not None:
    chi2 = np.sum(((mydata - ker_theo)/myerr)**2) / kpo1.kpi.nbkp
else:
    chi2 = np.sum(((mydata - ker_theo))**2) / kpo1.kpi.nbkp

print("test reduced chi2 = %.3f" % (chi2,))
