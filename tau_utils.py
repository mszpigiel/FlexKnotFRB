# tau_utils.py
import astropy.constants as aco
import astropy.units as aun
import cosmowrap as cw
import fgivenx
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import scipy.integrate as sin
import scipy.interpolate as sip
import scipy.optimize as sop
import scipy.stats as sst
from anesthetic.read.chain import read_chains
from anesthetic.samples import MCMCSamples, NestedSamples
from anesthetic.samples import merge_samples_weighted
#from anesthetic import read_chains
from copy import deepcopy


fid = deepcopy(cw.planck18_class_bestfit_ncdm)
fid['H0'] = 100*fid["h"]
fid['Omega_b_over_h'] = fid['omega_b']/fid['h']**3
fid['Omega_b'] = fid['omega_b']/fid['h']**2
fid['Omega_b_H0'] = fid['Omega_b']*fid['H0']
fid['Omega_m'] = (fid["omega_cdm"]+fid["omega_b"])/fid['h']**2

F = cw.cosmology(modules='newfrb', params=deepcopy(cw.planck18_class_bestfit_ncdm))
fast_tau_prefactor = F.toVal(F.n_H(0)*aco.c*aco.sigma_T/F.H(0), aun.one)*fid["h"]/fid["Omega_b"]
x_tmp, fast_tau_weights = np.polynomial.legendre.leggauss(999)
fast_tau_z_arr = 0+(50-0)/2*(x_tmp+1)
fast_tau_weights *= (50-0)/2

def fast_tau(Omega_b_over_h, Omega_m, *args, varyzend=True, usePCHIP=False):
    interp = sip.pchip if usePCHIP else sip.interp1d

    half = int(len(args)/2)
    zargs = [0, *args[:half+1], 50] if varyzend else [0, 5, *args[:half], 30, 50]
    xargs = [1, 1, *args[half+1:], 0, 0] if varyzend else [1, 1, *args[half:], 0, 0]

    custom_xifunc = interp(zargs, xargs)
    custom_xefunc = lambda z: F.xe1_of_xi(custom_xifunc(z)) + F.xeHeII_Planck(z)
    f = lambda z: custom_xefunc(z) * (1+z)**2 / np.sqrt(Omega_m*(1+z)**3 + (1-Omega_m))
    tau = fast_tau_prefactor * Omega_b_over_h * np.sum(f(fast_tau_z_arr) * fast_tau_weights)
    return float(tau)

def worker_call(params):
    # params = (Omega_b_over_h, Omega_m, args, varyzend, usePCHIP, consts_dict)
    Omega_b_over_h, Omega_m, args, varyzend, usePCHIP, consts = params
    return fast_tau(Omega_b_over_h, Omega_m, *args, varyzend=varyzend, usePCHIP=usePCHIP, **consts)