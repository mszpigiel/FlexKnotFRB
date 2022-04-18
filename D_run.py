import sys
from copy import deepcopy

import astropy.constants as aco
import astropy.units as aun
import numpy as np
import scipy.integrate as sin
import scipy.interpolate as sip
from cobaya.run import run

import cosmowrap as cw
from C_likelihood import *
from libflex import *


datafolder = 'data/'
chainfolder = 'chains/'

# This file is designed to run the Nested Sampling chains for our FRB analyses using
# command line arguments for the settings. Here is an example, the order does not matter:
# python3 D_run.py Mockdata_phys_sfr_z10%_norm100_obs1.npy planckpriors monotonous flexknot2 eprior varyzend hostcalx12
#  Mockdata_phys_sfr_z10%_norm100_obs1.npy is the file with synthetic data being loaded
#  planckpriors indicates to use Planck constraints for Omega_m and Omega_b_over_h
#  flexknot2 indicates using the FlexKnot algorhtm, and the number 2 gives the
#            "effective" number of knots, or the degrees of freedom divided by 2.
#            So "1" refers to moving start+end point, "2" adds one fully movable knot.
#  The 'monotonous' keyword enables enforcing a monotonous ionization history.
#  eprior indicates to use the common z*exp(-z) formulation for the prior on the FRB source distribution,
#         instead of the true prior distribution which might be unknown.
#  varyzend indicates moving the start and endpoints, otherwise they are fixed to redshifts 5 and 30, respectively.
#  I use the argument including 'host' (e.g. hostcalx12) to decide the output filename.

# Read parameters from command line arguments
#  Default values:
filename = "Mockdata_phys_sfr_z10%_norm100_obs1.npy" #data to read
earlytaulike = False # Whether to at the planck tau1530<0.006 1 sigma constraint
zmin = 0 # limit data to a above a certain zmin
zmax = 100 # limit data to a below a certain zmax
mode = "bayesian" # How to compute the likelihood (support only "bayesian" at the moment)
sampler = 'polychord' # Which sampler to use -- polychord or mcmc. Require polychord to compute evidences
priorsetting = None # Prior on Omega_b/h and Omega_m (flat, planck or delta)
flattentau = False # Whether to divide out implicity tau prior
usePCHIP = False # PCHIP vs linear interpolation between FlexKnots
use_relike = False #RELIKE likelihood to include CMB information

## Operational parameters
hostname = "host" # I run this code on various clusters, this is helpful for filenames
path = None #where to store the chains
likezoutput = False # Store the likes of each individual point
nlive = 0 # Set the number of live points (0=automatic)

## What method to use
flexknot = 0 # 0=tanh, otherwise flexknot
tanh_vary_deltaz = False # if using tanh whether to also vary \Delta z in addition to z_reio
DaiXia = False # Alternative parameters from Dai & Xia 2021
flexknot_monotonous = False #if flexknot whether to enforce monotonicity
varyzend = False #vary redshift of start and end points
flag_eprior = False #use e^-z prior in all cases

# Read command line arguments
print("Reading", sys.argv)
for arg in sys.argv:
    if arg[-3:]=='.py':
        print("  skipped reading", arg)
        continue
    if 'Mockdata' in arg:
        if 'None' in arg:
            filename = 'NoDataRun'
        else:
            filename = arg
    elif 'dz' in arg:
        tanh_vary_deltaz = True
    elif 'dai' in arg:
        DaiXia = True
        assert not tanh_vary_deltaz
    elif 'polychord'==arg or 'mcmc'==arg:
        sampler = arg
    elif 'chains' in arg:
        path = arg
    elif 'zmax' in arg:
        zmax=float(arg[4:])
    elif 'zmin' in arg:
        zmin=float(arg[4:])
    elif 'host' in arg:
        hostname = arg
    elif 'planckprior' in arg:
        priorsetting = "planck"
    elif 'planck5xprior' in arg:
        priorsetting = "5xplanck"
    elif 'flatprior' in arg:
        priorsetting = "flat"
    elif 'deltaprior' in arg:
        priorsetting = "delta"
    elif 'earlytaulike' in arg:
        earlytaulike = True
    elif 'relike' in arg:
        use_relike = True
        relike_l = relike.GaussianLikelihood()
        relike_p = relike.PC()
        relike_of_xe = lambda xe: relike_l.get_loglike(relike_p.get_mjs(xe))
    elif 'eprior' in arg:
        flag_eprior = True
    elif "flattentau" in arg:
        flattentau = True
    elif "likezoutput" in arg:
        likezoutput = True
    elif 'flexknot' in arg:
        flexknot = int(arg[8:])
    elif 'monotonous' in arg:
        flexknot_monotonous = True
    elif 'nlive' in arg:
        nlive = int(arg[5:])
    elif 'varyzend' in arg:
        varyzend = True
    elif "taski" in arg:
        task_index = int(arg[5:])
    elif "taskl" in arg:
        task_length = int(arg[5:])
    else:
        if not "run.py" in arg:
            assert False, "Could not parse argument"+str(arg)

# "half degrees of freedom", a less ambigous name for this variable
hdof = flexknot

# Load the data
if not filename == "NoDataRun":
    mockData = np.load(datafolder+filename, allow_pickle=True)
    print("Loaded", filename)
    N_points = len(mockData.item()['z_true'])
    print('Info:',  np.sum(mockData.item()['z_obs']>8), 'points at z> 8.')
    print('Info:',  np.sum(mockData.item()['z_obs']>10), 'points at z>10.')
    print('Info:', N_points, "total points.")
else:
    mockData = None

# Set up parameters for cobaya

## Basic cosmology
output_params = []
input_params = ["Omega_b_over_h", "Omega_m"]

## Reionization
extra_return_bins = [[0,5],[5,10],[10,15],[15,30]]
if not flexknot:
    if not DaiXia:
        # 2 parameter tanh model
        input_params.append("zreio")
        input_params.append('deltaz')
    else:
        # 2 parameter model from Dai & Xia 2020
        input_params.append("xi6")
        input_params.append('alpha')
else:
    assert not tanh_vary_deltaz, "flexknot is being used, 'dz' argument does not make sense."
    if not varyzend:
        for i in range(flexknot):
            input_params.append('z'+str(i+1))
            input_params.append('x'+str(i+1))
    else:
        for i in range(flexknot-1):
            input_params.append('z'+str(i+2))
            input_params.append('x'+str(i+2))
        input_params.append('z1')
        input_params.append('z'+str(flexknot+1))

# Extra output parameters
output_params.append('tau')
for bins in extra_return_bins:
    output_params.append('tau'+str(bins[0])+str(bins[1]))

# Extra likelihood outputs
output_params.append("logL_frb")
if earlytaulike:
    output_params.append("logL_tau1530")
if use_relike:
    assert not earlytaulike, "Using earlytaulike *and* relike is inconsistent!"
    output_params.append("logL_relike")
    output_params.append("logL_tau1530")

if flattentau:
    assert varyzend, "Currently only have prior fits for moving endpoints"
    output_params.append("logL_flattentau")
    fit_x = np.load("prior_runs/tau_interp.npy")
    fit_ys = []
    for i in range(flexknot):
        fit_ys.append(np.load("prior_runs/fit_hdof"+str(i+1)+"_planck_varyzend_linear_N1000000.npy"))
    assert np.allclose(1, sin.quad(sip.interp1d(fit_x, fit_ys[hdof-1]),0,1)[0], atol=0.01, rtol=0.01), ("Prior is not normalized", sin.quad(sip.interp1d(fit_x, fit_ys[0]),0,1))
else:
    fit_x = None
    fit_y = None

# Output params for individual likes
if likezoutput:
    for i in range(N_points):
        output_params.append('logL_z'+str(i))

# Distribution for z prior and dryrun tests
if "sfr" in filename:
    case = "sfr"
elif "exp" in filename:
    case = "exp"
elif "uni" in filename:
    case = "uni"
elif filename=="NoDataRun":
    case = "NoDataRun"
else:
    assert False, "Can't extract case (sfr, exp, uni) from filename"

# Set a default chains path
if path==None:
    path=chainfolder+hostname+'/run'

# Load cosmology Class, also computes Dispersion Measures
F = cw.cosmology(modules='data frb', params=deepcopy(cw.planck_lcdm_bestfit_class))

# Likelihood to pass to cobaya
def loglike_mcmc(flexknot=flexknot, case=case, **kwargs):
    # flexknot for half degrees of freedom, case for distribution

    # Cosmology parameters default:
    likelihood_kwargs = {
        "omegabh2": cw.planck_lcdm_bestfit_class['omega_b'],
        "omegach2": cw.planck_lcdm_bestfit_class['omega_cdm'],
        "H0": 100*cw.planck_lcdm_bestfit_class['h'],
        "logA": np.log(1e10*cw.planck_lcdm_bestfit_class['A_s']),
        "ns": cw.planck_lcdm_bestfit_class['n_s'],
        "mnu": 0.06, }

    # Background cosmo from kwargs
    Omega_b_over_h = kwargs.pop('Omega_b_over_h')
    Omega_m = kwargs.pop('Omega_m')
    h = likelihood_kwargs['H0']/100
    Omega_b = Omega_b_over_h*h
    Omega_c = Omega_m-Omega_b
    likelihood_kwargs["omegabh2"] = Omega_b*h**2
    likelihood_kwargs["omegach2"] = Omega_c*h**2

    # Reionization
    if not flexknot:
        if not DaiXia:
            zreio = kwargs.pop('zreio')
            deltaz = kwargs.pop('deltaz')
            likelihood_kwargs['custom_xifunc'] = F.xifunc_Planck(zreio1=zreio, dz=deltaz)
        else:
            xi6 = kwargs.pop("xi6")
            alpha = kwargs.pop("alpha")
            likelihood_kwargs['custom_xifunc'] = lambda z, xi6=xi6, alpha=alpha: np.heaviside(6-z,1)*(1-(1-xi6)*(1+z)**3/(1+6)**3)+np.heaviside(z-6,0)*xi6*np.exp(alpha*(6-z))
    else:
        likelihood_kwargs['custom_xifunc'], kwargs = flexknots_to_function(kwargs,
                          interp_method=scipy.interpolate.interp1d, debug=False,
                          min_pos=5, max_pos=30, left_val=1, right_val=0,
                          move_endpoints=None, n_params=None, pos='z', val='x',
                          return_pop_kwargs=True, return_hdof=False)

    # kwargs should be empty now
        assert kwargs=={}, "kwargs should be empty now - did you forget something? "+str(kwargs.keys())


    # Use exponential prior instead of the (possibly unknown) true distribution
    if flag_eprior:
        likelihood_kwargs["zprior"] = lambda z, case="exp": F.frb_distribution(np.asarray(z), case=case)
    else:
        likelihood_kwargs["zprior"] = lambda z, case=case: F.frb_distribution(np.asarray(z), case=case)

    if use_relike:
        likelihood_kwargs["relike"] = relike_of_xe
    else:
        likelihood_kwargs["relike"] = None

    # kwargs that are not passed because fixed over MCMC run or hardcoded
    likelihood_kwargs = {**likelihood_kwargs,
        "zmin": zmin,
        "zmax": zmax,
        "earlytaulike": earlytaulike,
        "mockData": mockData,
        "cosmowrapClass": F,
        "mode": mode,
        "integration_method": "leggauss",
        "extra_return_bins": extra_return_bins,
        "flattentau": flattentau,
        "indiv_logL_output": likezoutput,
        "debugplot": False,
        "zmin_bayesian": 2,
        "zmax_bayesian": 24,
        "zmax_reiosample": 50,
        "integration_steps": 999,
    }
    if flattentau:
        likelihood_kwargs["fit_x"] = fit_x
        likelihood_kwargs["fit_y"] = fit_ys[hdof-1]
    
    return loglike_full(**likelihood_kwargs)


# Now write a dictionary for cobaya sampling:

infodict = {
    'debug': False,
    'timing': True,
    'verbose': True,
    'seed': 42,
    'resume': True,
    "sampler":{
        sampler:{
        }
    },
    "likelihood":{
        "mylike": {
            "external": loglike_mcmc,
            "input_params": input_params,
            "output_params": output_params,
            }
    }
}

# nlive==0 implies using the default number (determined by cobaya from dimensionality)
if nlive>0:
    infodict['sampler'][sampler]['nlive'] = nlive

# Baseline parameters, flat priors
infodict["params"] = {
    "Omega_b_over_h": {"prior": {"dist": "uniform", "min": 0.02, "max": 0.10}, "ref": {"dist": "norm", "loc": 0.02242/0.6766**3,  "scale": 0.00002}, "latex": r"\Omega_\mathrm{b} / h"},
    "Omega_m": {"prior": {"dist": "uniform", "min": 0,    "max": 1},      "ref": {"dist": "norm", "loc": (0.11933+0.02242)/0.6766**2,  "scale": 0.0002}, "latex": r"\Omega_\mathrm{m}"},
    "zreio": {"prior": {"dist": "uniform", "min": 5, "max": 15},    "ref": {"dist": "norm", "loc": 7.82,   "scale": 0.71}, "latex": r"\tau"},
}
# latex labels for extra outputs
if likezoutput:
    for i in range(N_points):
        infodict["params"]['logL_z'+str(i)] = {"latex": r"L{0:}".format(i)}
infodict["params"]['logL_frb'] = {"latex": r"L_{\rm FRBs}"}
if earlytaulike:
    infodict["params"]['logL_tau1530'] = {"latex": r"L_{\rm earlytau}"}
if use_relike:
    infodict["params"]['logL_relike'] = {"latex": r"L_{\rm RELIKE}"}
    infodict["params"]['logL_tau1530'] = {"latex": r"L_{\rm earlytau}"}

# Planck priors
if priorsetting == 'planck':
    infodict["params"]["Omega_b_over_h"] = {"prior": {"dist": "norm", "loc": 0.0724,  "scale": 0.0011}, "latex": r"\Omega_\mathrm{b}/h"}
    infodict["params"]["Omega_m"] = {"prior": {"dist": "norm", "loc": 0.31108,  "scale": 0.00555}, "latex": r"\Omega_\mathrm{m}"}
    planckstring='_pprior'
elif priorsetting == '5xplanck':
    infodict["params"]["Omega_b_over_h"] = {"prior": {"dist": "norm", "loc": 0.0724,  "scale": 0.0011*5}, "latex": r"\Omega_\mathrm{b}/h"}
    infodict["params"]["Omega_m"] = {"prior": {"dist": "norm", "loc": 0.31108,  "scale": 0.00555*5}, "latex": r"\Omega_\mathrm{m}"}
    planckstring='_5xpprior'
# Delta function priors
elif priorsetting == 'delta':
    infodict["params"]["Omega_b_over_h"] = {"value": 0.0724, "latex": r"\Omega_\mathrm{b}/h"}
    infodict["params"]["Omega_m"] = {"value": 0.31108, "latex": r"\Omega_\mathrm{m}"}
    planckstring='_dprior'
else:
    planckstring='_fprior'

# Output strings to indicate what was used
if earlytaulike:
    planckstring+='_etl'
if use_relike:
    planckstring+='_relike'
if flag_eprior:
    planckstring+='_eprior'
if usePCHIP:
    planckstring+='_pchip'
if likezoutput:
    planckstring+="_likez"
if varyzend:
    planckstring+="_vzend"

infodict['params']['tau'] = {"latex": r"\tau"}
for bins in extra_return_bins:
    infodict['params']['tau'+str(bins[0])+str(bins[1])] = {"latex": r"\tau_{{{0:}, {1:}}}".format(str(bins[0]), str(bins[1]))}

if flexknot:
    if flexknot_monotonous:
        dzstring = '_monotonousflexknot'+str(flexknot)
    else:
        dzstring = '_flexknot'+str(flexknot)
    if flattentau:
        dzstring+='_flattau'
        if not mode=="dummyflex":
            infodict['params']['logL_flattentau'] = {"latex": r"$\log L_{\rm flattentau}$"}
    infodict['params'].pop("zreio")
    # This function samples from ordered numbers as in 1506.00171, note A13 should be
    # theta_i = theta_{i-1} + (theta_max-theta_{i-1}) (1-x_i^(1/(n-i+1))) 
    flexknot_params = flexknot_cobaya_params(n_params=2*flexknot, move_endpoints=varyzend,
                        monotonous=flexknot_monotonous, xmin=5, xmax=30, yleft=1, yright=0)
    infodict['params'].update(flexknot_params)
elif tanh_vary_deltaz:
    # vary \Delta z in addition to z_reio
    dzstring = '_vardz'
    infodict['params']["deltaz"] = {"prior": {"dist": "uniform", "min": 0.1, "max": 3},    "ref": {"dist": "norm", "loc": 0.5,   "scale": 0.1}, "latex": r"\Delta z"}
else:
    # fix \Delta z = 0.5 (default)
    infodict['params']["deltaz"] = {"value": 0.5, "latex": r"\Delta z"}
    dzstring = '_defdz'

if DaiXia:
    # use a different 2-parameter model from Dai & Xia 2020
    infodict['params'].pop("zreio")
    infodict['params'].pop("deltaz")
    infodict['params']["xi6"] = {"prior": {"dist": "uniform", "min": 0.9, "max": 1}, "latex": r"x_{i,6}"}
    infodict['params']["alpha"] = {"prior": {"dist": "uniform", "min": 0, "max": 2}, "latex": r"\alpha"}
    dzstring = '_dai'

# Put all the strings together to an output filename
infodict['output'] = path+'_'+hostname+'_'+sampler[0:4]+mode[0:2]+'_'+filename[:-4]+dzstring+planckstring

# Add mask information if used (almost never)
if zmin != 0:
    infodict['output'] += '_zmin='+str(zmin)
if zmax != 100:
    infodict['output'] += '_zmax='+str(zmax)


print("Infodict:", infodict)
for key in infodict['params']:
    print(key, ":", infodict['params'][key])



run(infodict)
