import astropy.units as aun
import numpy as np
import scipy.interpolate as sip
import scipy.stats as sst

import cosmowrap as cw

# Likelihood function of mockData given:
#  tmp_cosmology -- Class that computes cosmology stuff
#  integration_method, integration_steps -- details how to marginalize over z
#  zprior,zmin_bayesian,zmax_bayesian -- prior for redshift, and its bounds
#  zmin, zmax -- for masking dat beyond a certain redshift range
#  custom_xifunc -- the reionization history to consider
#  indiv_logL_output -- a flag to output the logL of every individual data point, in addition to the sum
def loglike_frb(mockData,tmp_cosmology,integration_method,integration_steps,zprior,zmin_bayesian,zmax_bayesian,zmin,zmax,custom_xifunc,indiv_logL_output):
    assert zprior!=None, "Need prior redshift distribution zprior"

    # Load observations from mockData dictionary
    z_obs = np.array(mockData['z_obs'])
    z_err_rel = mockData['settings']['z_error_relative']
    DM_obs = np.array(mockData['DM_obs'])
    DM_nonCosmo = mockData['settings']['DM_offset']
    sigma_DM_MW = mockData['settings']['sigma_DM_MW']
    sigma_DM_host0 = mockData['settings']['sigma_DM_host0']

    # Mask regions if we filter if lowz/highz
    ge = np.greater_equal(z_obs, zmin)
    l = np.less(z_obs, zmax)
    bool_mask = np.logical_and(ge, l)
    mask = np.where(bool_mask)
    if not np.all(bool_mask):
        print("Masking", np.sum(np.logical_not(bool_mask)), "data points:", mask)

    # Compute FRB likelihood
    #  Integral preparation
    print("Using", integration_method, "integration.")
    if integration_method=="simple":
        z_arr = np.linspace(zmin_bayesian,zmax_bayesian,integration_steps)
        weights = (zmax_bayesian-zmin_bayesian)/integration_steps
    elif integration_method=="leggauss":
        x, weights = np.polynomial.legendre.leggauss(integration_steps)
        z_arr = zmin_bayesian+(zmax_bayesian-zmin_bayesian)/2*(x+1)
        weights *= (zmax_bayesian-zmin_bayesian)/2
    else:
        assert False, "Unknown integration_method"

    zerr_arr = z_err_rel*z_arr
    if np.all(zerr_arr!=0):
        DM_model_arr = DM_nonCosmo+tmp_cosmology.toVal(tmp_cosmology.fast_DM_array(z_arr, xi_func=custom_xifunc), aun.pc/aun.cm**3)
        # Note: The following is a minor approximation since sigmaDM_obs_of_z_and_DM depends on "DM_earth", i.e. DM_model plus scatter.
        #       However this dependence via dispersion smearing is small and sigmaDM_obs also small, so we can use DM_model instead.
        #       This saves us a 2nd integral which hugely accelerates the computation
        sigma_DM_IGM = tmp_cosmology.toVal(tmp_cosmology.sigma_DM_IGM_function(xi_func=custom_xifunc, zmin=np.min(z_arr), zmax=np.max(z_arr))(z_arr), aun.pc/aun.cm**3)
        # Currently only considering the normal observational error
        obserror_factor = 1
        sigma_DM_obs = tmp_cosmology.toVal(tmp_cosmology.sigmaDM_obs_of_z_and_DM(z_arr, DM_model_arr*aun.pc/aun.cm**3, calibration_factor=obserror_factor), aun.pc/aun.cm**3)
        DM_err_arr = np.sqrt(sigma_DM_MW**2+(sigma_DM_host0/(1+z_arr))**2+sigma_DM_obs**2+sigma_DM_IGM**2)
        # Do those N integrals (for N data points) in parallel:
        Z = np.tile(z_obs, (integration_steps, 1)).T
        DM = np.tile(DM_obs, (integration_steps, 1)).T
        result = sst.norm.pdf(Z, loc=z_arr, scale=zerr_arr)*sst.norm.pdf(DM, loc=DM_model_arr, scale=DM_err_arr)*zprior(z_arr)
        individual_likes_per_obs_point = np.sum(result*weights, axis=1)
    else:
        assert False, "Perfect redshift measurements not implemented at the moment."

    # Resulting logL for all points combined:
    logL_frb = np.sum(np.log(individual_likes_per_obs_point[mask]), axis=0)
    assert not np.isnan(logL_frb), "Error (NaN result) in the integration"
    if not indiv_logL_output:
        return logL_frb
    else:
        return logL_frb, individual_likes_per_obs_point

# Likelihood with all parameters as kwargs, including more than just FRBs
def loglike_full(
    omegabh2=cw.planck_lcdm_bestfit_class['omega_b'], omegach2=cw.planck_lcdm_bestfit_class['omega_cdm'], H0=100*cw.planck_lcdm_bestfit_class['h'],
    tau=cw.planck_lcdm_bestfit_class['tau_reio'], logA=np.log(1e10*cw.planck_lcdm_bestfit_class['A_s']), ns=cw.planck_lcdm_bestfit_class['n_s'],
    mnu=0.06, dz=False, debugplot=False, zmin=0, zmax=100, custom_xifunc=None, zmax_reiosample=50, earlytaulike=False,
    mockData=None, cosmowrapClass=None, mode="bayesian", integration_steps=999, integration_method="leggauss", zprior=None, extra_return_bins=None,
    zmin_bayesian=2, zmax_bayesian=24, clikpath="/home/stefan/.links/planck_data/", printparams=False,
    flattentau=False, fit_x=None, fit_y=None, indiv_logL_output=False, fast_tau=None, newcode=True, relike=None, just_tau_prior_run=False):
    # zmin, zmax: Mask for which points to use
    # zmin_bayesian, zmax_bayesian: Limits for z_true in Bayesian integration, defaults to 2 and 24 which is 3 sigma away from 5,15 @20% z error
    #                               and is accurate in loglike to +/- ~1e-4
    # flattentau: "Flatten" the prior on tau by modifying the prior according to
    # Handley & Millea https://arxiv.org/abs/1804.08143
    # Some checks
    if mockData is None:
        print("No mockData passed! Assuming prior-run")
        just_tau_prior_run=True

    if custom_xifunc != None:
        assert extra_return_bins!=None, "Please pass extra_return_bins as well"

    assert relike is None, "RELIKE calls currently not implemented (assume x_i==1 for z<6)"

    # Total loglikelihood returned
    total_loglike = 0
    ## Extra outputs:
    extra_return_dict = {}
    ### Extra likelihoods
    extra_return_dict['logL_frb'] = 0
    if earlytaulike:
        extra_return_dict['logL_tau1530'] = 0
    if flattentau:
        extra_return_dict['logL_flattentau'] = 0
    if relike is not None:
        extra_return_dict['logL_relike'] = 0
    ### Extra parameters
    if custom_xifunc != None:
        extra_return_dict['tau'] = 0
    if custom_xifunc != None:
        for bins in extra_return_bins:
            extra_return_dict['tau'+str(bins[0])+str(bins[1])] = 0

    # If out-of-parameter limits or code crash return -np.inf
    def failureExit(reason):
        print("Returning failure due to "+reason+"!")
        return -np.inf, extra_return_dict

    # Avoid CMB code CLASS to crash
    if tau<4e-3:
        return failureExit("Tau too small for CLASS!")

    # Use astropy for cosmology
    params = {'omega_b': omegabh2, 'omega_cdm': omegach2, 'h': H0/100}
    tmp_cosmology = cw.cosmology(modules='frb', params=params)
    tmp_tau_reio = tmp_cosmology.optical_depth_of_xi(custom_xifunc)
    extra_return_dict['tau'] = tmp_tau_reio

    # Extra outputs, optical depth in between extra_return_bins
    for bins in extra_return_bins:
        extra_return_dict['tau'+str(bins[0])+str(bins[1])] = tmp_cosmology.optical_depth_of_xi(custom_xifunc, zlow=bins[0], zhigh=bins[1], integrate_mode=1e5)

    # In case of no-data run
    if just_tau_prior_run:
        if flattentau:
            if tmp_tau_reio<0.3 and tmp_tau_reio>0.03:
                taufit = sip.interp1d(fit_x, fit_y)(tmp_tau_reio)
                logprior_taufit = np.log(taufit)
                logL_flattentau = -logprior_taufit
            else:
                logL_flattentau = -1e3
            return logL_flattentau, extra_return_dict
        else:
            return 0, extra_return_dict

    mockData = mockData.item()
    ### Detailled outputs:
    if indiv_logL_output:
        for i in range(len(np.array(mockData['z_true']))):
            extra_return_dict["logL_z"+str(i)] = 0

    # FRB Likelihood, reading data
    if not indiv_logL_output:
        logL_frb = loglike_frb(mockData,tmp_cosmology,integration_method,integration_steps,zprior,zmin_bayesian,zmax_bayesian,zmin,zmax,custom_xifunc,indiv_logL_output)
    else:
        logL_frb, individual_likes_per_obs_point = loglike_frb(mockData,tmp_cosmology,integration_method,integration_steps,zprior,zmin_bayesian,zmax_bayesian,zmin,zmax,custom_xifunc,indiv_logL_output)
        # Return individual likelihoods. Convention: Return numbers for FRBs sorted by observed source redshift.
        z_obs = np.array(mockData['z_obs'])
        order = np.argsort(z_obs)
        for i in range(len(order)):
            extra_return_dict["logL_z"+str(i)] = np.log(individual_likes_per_obs_point[order[i]])

    extra_return_dict['logL_frb'] = logL_frb
    total_loglike += logL_frb


    # Extra likelihoods
    if earlytaulike:
        tau1530 = tmp_cosmology.optical_depth_of_xi(custom_xifunc, zlow=15, zhigh=30)
        logL_tau1530 = tmp_cosmology.planck_earlytau_loglike(tau1530)
        extra_return_dict['logL_tau1530'] = logL_tau1530
        total_loglike += logL_tau1530
    if relike is not None:
        logL_relike = relike(tmp_cosmology.xefunc_of_xifunc(custom_xifunc))
        extra_return_dict['logL_relike'] = logL_relike
        total_loglike += logL_relike
        tau1530 = tmp_cosmology.optical_depth_of_xi(custom_xifunc, zlow=15, zhigh=30)
        logL_tau1530 = tmp_cosmology.planck_earlytau_loglike(tau1530)
        extra_return_dict['logL_tau1530'] = logL_tau1530
    if flattentau:
        if tmp_tau_reio<0.3 and tmp_tau_reio>0.03:
            taufit = sip.interp1d(fit_x, fit_y)(tmp_tau_reio)
            logprior_taufit = np.log(taufit)
            logL_flattentau = -logprior_taufit
        else:
            logL_flattentau = -1e3
        extra_return_dict['logL_flattentau'] = logL_flattentau
        total_loglike += logL_flattentau

    # Return message
    print('Returning {0:.2f}'.format(total_loglike)+' with logL_frb={0:.2f}'.format(logL_frb), " @ {0:.4f}".format(tmp_tau_reio))
    if total_loglike<-1e10:
        print("Very very low logL!")
        for key in extra_return_dict.keys():
            print(key, ":", extra_return_dict[key])

    return total_loglike, extra_return_dict
