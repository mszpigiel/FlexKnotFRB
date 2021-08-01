import astropy.units as aun
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as sip
from copy import deepcopy

import cosmowrap as cw


# Setup what to generate
#  Data sets with 100, 1,000 and 10,000 FRBs
normalizations = [100, 1000, 10000]
#  FRB source distribution scaling with Star Formation Rate
#  (also implemented uniform distribution and z*exp(-z) exponential distribution)
cases = ['sfr']
#  Seeds: Make sure to have a different seed for every new combination,
#         but make sure than the 1,000 FRB data set includes the 100
#         FRB data set, i.e. same seed there.
seed_case = {'exp': 1, 'sfr': 2, 'uni': 3}
models = {'phys': 20}  # {'tanh':10, 'phys':20}
# Scale the redshift and observational uncertainties.
#  Redshift is relative error
redshift_error_configs = [0.1]  # [0, 0.01, 0.1, 0.2]
#  Obs. error == 1 scales from FRB180924 (accounting for higher/lower redshift)
observation_error_configs = [1]  # [0.1, 1, "perfect"]
#  Note: Current likelihood assumes observation_error_configs==1

# Other dispersion measure contributions and uncertainties
#  Milky Way disk
DM_MW_disk = 500
sigma_DM_MW_disk = 20
#  Milky Way halo
DM_MW_halo = 50
sigma_DM_MW_halo = 50
#  Host galaxy/environment contribution
DM_host0 = 200
sigma_DM_host0 = 100
#  Milky way combined error
sigma_DM_MW = np.sqrt(sigma_DM_MW_disk**2+sigma_DM_MW_halo**2)
#  Combined mean contributions
#  Note: DM_host0 should be scaled with redshift. We don't
#        do this here as the DM_nonCosmo contribution only serves
#        to contribute to a realistic estimate of the dispersion
#        smearing error and is subtracted later anyway
DM_nonCosmo = DM_MW_disk + DM_MW_halo + DM_host0

# Where to load & store the data
datafolder = 'data/'

params = deepcopy(cw.planck18_class_bestfit_ncdm)
params["extended_data"] = True
C = cw.cosmology(modules='data frb', params=params)


def load_reionization_history(filename):
    reio_history = np.genfromtxt(filename).T
    zreio =list(reio_history[0])
    xreio = list(reio_history[1])
    logreio_history_interp = sip.interp1d(zreio, np.log(xreio), kind="linear", fill_value="extrapolate")
    linreio_history_interp = sip.interp1d([14, 15], [np.exp(logreio_history_interp(14)), 0], kind="linear", fill_value="extrapolate")
    return lambda z: np.minimum(1, np.maximum(0, np.exp(logreio_history_interp(z))*np.heaviside(14-z, 0)+linreio_history_interp(z)*np.heaviside(z-14, 1)))

# Kulkarni 2019 et al. (https://arxiv.org/abs/1809.06374)
xi_phys = load_reionization_history("kulkarni_et_al_2019_reionization.txt")


# Generate synthetic observations
print('Generating measurements with all the errors')
for model in models:
    for norm in normalizations:
        for case in cases:
            # Load data from previous code
            filename_in = model+"_samples_"+case+"_norm"+str(norm)+'.npy'
            DM_measurements = np.load(datafolder+filename_in, allow_pickle=True)
            # Iterate through error configurations
            for obserror_factor in observation_error_configs:
                assert obserror_factor==1, "Currently only observation_error_configs == 1 supported in likelihood"
                for z_error_relative in redshift_error_configs:
                    filename_out = 'Mockdata_'+model+'_{0:}_z{1:.0f}%_norm{2:}_obs{3:}.npy'.format(case, z_error_relative*100, norm, obserror_factor)
                    output_dict = {'z_true':[], 'DM_meancosmo':[],
                                   'z_obs':[], 'DM_obs':[], 'DM_err':[],
                                   'settings': {}}
                    output_dict['settings'] = {'z_error_relative': z_error_relative, 'DM_offset': DM_nonCosmo, 'sigma_DM_MW': sigma_DM_MW, "sigma_DM_host0": sigma_DM_host0}

                    # Use this to only create new seeds for a new distributions and mode,
                    # keep same seed if just increasing data set size
                    seed = seed_case[case]+models[model]
                    np.random.seed(seed)
                    # Return IGM scatter DM uncertainty as a function of redshift
                    sigma_DM_IGM_func = C.sigma_DM_IGM_function(xi_func=xi_phys)
                    for i in range(len(DM_measurements)):
                        sample = DM_measurements[i]
                        # Roll random numbers first to avoid messing up things,
                        # e.g. when having no redshift error
                        zrand = np.random.normal(loc=0, scale=1)
                        DM1rand = np.random.normal(loc=0, scale=1)
                        DM2rand = np.random.normal(loc=0, scale=1)
                        DM3rand = np.random.normal(loc=0, scale=1)
                        # True values (expectation value / mean IGM value)
                        z = sample[0]
                        DM_cosmo = sample[1]
                        output_dict['z_true'].append(z)
                        output_dict['DM_meancosmo'].append(DM_cosmo)
                        # Realistic redshift error (e.g. 10%)
                        z_err = z*z_error_relative
                        z_obs = z + zrand*z_err
                        output_dict['z_obs'].append(z_obs)
                        # Dispersion Measure uncertainty, all quantities in units of pc/cm^3
                        sigma_DM_IGM = C.toVal(sigma_DM_IGM_func(z), aun.pc/aun.cm**3)
                        DM_err_nonObs = np.sqrt(sigma_DM_IGM**2+sigma_DM_MW_disk**2+sigma_DM_MW_halo**2+(sigma_DM_host0/(1+z))**2)
                        DM_real = DM_cosmo+DM_nonCosmo+DM1rand*DM_err_nonObs
                        # "Real" corresponds to the DM of the burst when it arrives, now we add the observational error
                        # which (very slightly) depends on DM_real (dispersion smearing)
                        DMobserr = 0 if obserror_factor=="perfect" else C.toVal(C.sigmaDM_obs_of_z_and_DM(z, DM_real*aun.pc/aun.cm**3, calibration_factor=obserror_factor), aun.pc/aun.cm**3)
                        DM_obs =  DM_real + DM2rand*DMobserr
                        output_dict['DM_obs'].append(DM_obs)
                        output_dict['DM_err'].append(np.sqrt(DMobserr**2+DM_err_nonObs**2))
                        # If this would happen we would have to take into account that our PDF is cut-off at z<0, but should not occur
                        assert z_obs>0, "z<0 cut-off not treated yet"
                        assert DM_obs>0, "DM<0 cut-off not treated yet"
                    
                    # Sort data, just for convinience in some plots, not relevant for likelihood
                    order = np.argsort(output_dict['z_true'])
                    for key in output_dict.keys():
                        if type(output_dict[key]) is list:
                            output_dict[key] = np.array(output_dict[key])[order]
                    
                    np.save(datafolder+filename_out, output_dict)
                    print("Saved to", datafolder+filename_out)




def plot_data_generation(d):
    plt.plot(d["z_true"], d['DM_meancosmo']+DM_nonCosmo, label='Modelled true DM and z', marker='o', color='green')
    plt.scatter(d["z_obs"], d['DM_obs'], marker='o', label='Observed data', color='orange')
    for i in range(len(d['z_true'])):
        plt.plot([d['z_true'][i],d['z_true'][i],d['z_obs'][i]], [d["DM_meancosmo"][i]+DM_nonCosmo,d["DM_obs"][i],d["DM_obs"][i]], lw=0.5)
    plt.legend(loc='lower right')
    plt.xlabel('Redshift $z$')
    plt.ylabel('Dispersion measure DM [pc/cm3]')
    plt.show()

def plot_errorbars(d):
    style = {"fmt":'o', "markersize":4, "markerfacecolor":'orange', "markeredgecolor":'black'}
    plt.errorbar(d["z_obs"], d['DM_obs'], yerr=d['DM_err'], xerr=0.1*d["z_true"], **style, label='Synthetic data')
    plt.plot(d["z_true"], d['DM_meancosmo']+DM_nonCosmo, label=r'True $\overline{DM}(z)$', color='green')
    plt.legend(loc='lower right')
    plt.xlabel('Redshift $z$')
    plt.ylabel(r'Dispersion measure DM $[\mathrm{pc\,cm^{-3}}]$')
    plt.show()

d = np.load(datafolder+"Mockdata_phys_sfr_z10%_norm100_obs1.npy", allow_pickle=True).item()

plot_data_generation(d)
plot_errorbars(d)