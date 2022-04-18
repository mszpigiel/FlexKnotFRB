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
models = ['phys']
#  Seeds: Make sure to have a different seed for every new combination,
#         but make sure than the 1,000 FRB data set includes the 100
#         FRB data set, i.e. want to use the same seed there.
#  Additional runs with new seed:
#  (1) new DM values, same redshift (add_seed=1)
#  (2) new z values (B), and new DM values (add_seed=2)
#  (3) new z values (C), and new DM values (add_seed=3)
add_seed = 0 # add to regenerate DM data
data_samples = "samplesA"
seed_case = {'sfr': 2} # {'exp': 1, 'sfr': 2, 'uni': 3}
seed_models = {'phys': 20}  # {'tanh':10, 'phys':20}

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
            filename_in = model + "_" + data_samples + "_" + case + '_norm' + str(norm) + '.npy'
            DM_measurements = np.load(datafolder+filename_in, allow_pickle=True)
            # Iterate through error configurations
            for obserror_factor in observation_error_configs:
                assert obserror_factor==1, "Currently only observation_error_configs == 1 supported in likelihood"
                for z_error_relative in redshift_error_configs:
                    filename_out = 'Mockdata_'+model+'_seed'+str(add_seed)+'_{0:}_z{1:.0f}%_norm{2:}_obs{3:}.npy'.format(case, z_error_relative*100, norm, obserror_factor)
                    output_dict = {'z_true':[], 'DM_meancosmo':[],
                                   'z_obs':[], 'DM_obs':[], 'DM_err':[],
                                   'settings': {}}
                    output_dict['settings'] = {'z_error_relative': z_error_relative, 'DM_offset': DM_nonCosmo, 'sigma_DM_MW': sigma_DM_MW, "sigma_DM_host0": sigma_DM_host0}

                    # Use this to only create new seeds for a new distributions and mode,
                    # keep same seed if just increasing data set size
                    seed = seed_case[case]+seed_models[model]+add_seed
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



# Some plots to illustrate (i) steps how the data is generate, and (ii) the final sample of synthetic data

def plot_data_generation(d):
    plt.figure()
    plt.plot(d["z_true"], d['DM_meancosmo']+DM_nonCosmo, label='Modelled true DM and z', marker='o', color='green')
    plt.scatter(d["z_obs"], d['DM_obs'], marker='o', label='Observed data', color='orange')
    for i in range(len(d['z_true'])):
        plt.plot([d['z_true'][i],d['z_true'][i],d['z_obs'][i]], [d["DM_meancosmo"][i]+DM_nonCosmo,d["DM_obs"][i],d["DM_obs"][i]], lw=0.5)
    plt.legend(loc='lower right')
    plt.xlabel('Redshift $z$')
    plt.ylabel('Dispersion measure DM [pc/cm3]')

def plot_errorbars(d):
    plt.figure()
    style = {"fmt":'o', "markersize":4, "markerfacecolor":'orange', "markeredgecolor":'black'}
    plt.errorbar(d["z_obs"], d['DM_obs'], yerr=d['DM_err'], xerr=0.1*d["z_true"], **style, label='Synthetic data')
    plt.plot(d["z_true"], d['DM_meancosmo']+DM_nonCosmo, label=r'True $\overline{DM}(z)$', color='green')
    plt.legend(loc='lower right')
    plt.xlabel('Redshift $z$')
    plt.ylabel(r'Dispersion measure DM $[\mathrm{pc\,cm^{-3}}]$')

filename_plot = 'Mockdata_'+model+'_seed'+str(add_seed)+'_{0:}_z{1:.0f}%_norm{2:}_obs{3:}.npy'.format("sfr", 0.1*100, 100, 1)
d = np.load(datafolder+filename_plot, allow_pickle=True).item()

plot_data_generation(d)
plt.savefig("extra_plots/B_data_generation.pdf")
plot_errorbars(d)
plt.savefig("extra_plots/B_errorbars.pdf")
plt.show()


# Double check that files which should be identical indeed are identical
# (just use a single number to check the random seeds are right)

print("Check for data files:")
import glob
print("# z samples -- The legacy one (no letter) should be identical to samplesA.")
print("# samplesB and samplesC should have different z_true and DM_meancosmo samples.")
print("# {0:<54} {1: <7} {2: <7}".format("Filename", "z_true", "DM_meancosmo"))
for norm in normalizations:
    print("###", norm, "point files:")
    files = glob.glob("./data/phys_samples*_norm{0:}.npy".format(norm))
    for f in files:
        file_content = np.load(f, allow_pickle=True).T
        print("# {0:<54} {1:.4f} {2:.4f}".format(f, np.sort(file_content[0])[-1], np.sort(file_content[1])[-1]))

print("# DM samples -- The legacy one (no seed number) should be identical to 0.")
print("# The seed1 one should have the same z but different DM sample (DM_obs).")
print("# The seed2 and seed3 ones should be completely different.")
print("# {0:<54} {1: <7} {2: <7} {3: <9} {4: <9}".format("Filename", "z_true", "z_obs", "DM_meanc.", "DM_obs"))
for norm in normalizations:
    print("###", norm, "point files:")
    files = glob.glob("./data/Mockdata_*_norm{0:}_obs1.npy".format(norm))
    for f in files:
        file_content = np.load(f, allow_pickle=True).item()
        print("# {0:<54} {1:.4f} {2:.4f} {3:.4f} {4:.4f}".format(f, np.sort(file_content["z_true"])[-1], np.sort(file_content["z_obs"])[-1], np.sort(file_content["DM_meancosmo"])[-1], np.sort(file_content["DM_obs"])[-1]))
