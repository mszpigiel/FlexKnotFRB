import astropy.constants as aco
import astropy.units as aun
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sin
import scipy.interpolate as sip
import seaborn as sns
colors = sns.color_palette("colorblind")
from copy import deepcopy

import cosmowrap as cw



# Setup what to generate
#  Data sets with 100, 1,000 and 10,000 FRBs
normalizations = [100, 1000, 10000]
#  FRB source distribution scaling with Star Formation Rate
#  (also implemented uniform distribution and z*exp(-z) exponential distribution)
cases = ['sfr']
shift_seed = 0
data_samples = "samplesA"
# samplesB and samplesC with seed shifted by 1 and 2 respectively
# Just "samples" are the old data, identical to samplesA
# (though npy file differs due to newer software versions)

# Load the base cosmological model, using my cosmology codes wrapper

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

# Compute how many FRBs we should have between redshifts 5-8, 8-10, 10-15
#  Separating these intervals makes the accept-reject sampling below more efficient
phases = {}
for i in range(len(cases)):
    case = cases[i]
    phases[case] = {}
    f = lambda z: C.frb_distribution(z, case=case)
    phases[case]["5-8"] = {"config": [5, 8, sin.quad(f, 5, 8, epsrel=1e-4)[0]]}
    phases[case]["8-10"] = {"config": [8, 10, sin.quad(f, 8, 10, epsrel=1e-4)[0]]}
    phases[case]["10-15"] = {"config": [10, 15, sin.quad(f, 10, 15, epsrel=1e-3, limlst=200)[0]]}
    print("Samples in each redshift segment (segments for efficient rejection-sampling):")
    print(case, "Total:", sin.quad(f, 5, 15, epsrel=1e-5)[0], "=", np.sum([phases[case]["5-8"]['config'][2], phases[case]["8-10"]['config'][2], phases[case]["10-15"]['config'][2]]))
    print(case, "5-8:", phases[case]["5-8"]['config'][2])
    print(case, "8-10:", phases[case]["8-10"]['config'][2])
    print(case, "10-15:", phases[case]["10-15"]['config'][2])

# Generating FRB mock daya
#  Could have multiple "models" here, e.g. using the physical reionization history by Kulkarni et al. 2019 or a simple tanh one
models = {'phys': C}
#  Dictionary storing the data (uniform and exponential distributions not used in this version)
data = {'uni': {'samples': []}, 'exp': {'samples': []}, 'sfr': {'samples': []}}
#  Set a constant seed for reproducibility
np.random.seed(42+shift_seed)
for number_total in normalizations:
    print("-- Generating N =", number_total, "samples --")
    for case in phases.keys():
        highz_targetsum = 0
        # Calculate how many samples to generate per bin, starting with the high redshift bins
        steps = np.array(list(phases[case].keys()))[::-1]
        for step in steps:
            # lower bound, upper bound
            lb, ub, target = phases[case][step]["config"]
            if not step == '5-8':
                target = round(target * number_total)
                highz_targetsum += target
            else:
                # In the last (and biggest, z = 5 .. 8) step account for rounding errors, minor change
                print("Original target:", target * number_total, "changed to", number_total - highz_targetsum, "to account for rounding.")
                target = number_total - highz_targetsum
            # Do a very simple accept-reject sampling, no need to optimize further since fast and run only once usually
            maxdNdz = np.max(C.frb_distribution(np.linspace(lb, ub, 10000), case=case))
            print("Sampling", target, "in step", step, "with", case, "distribution.")
            while np.sum(np.logical_and(np.array(data[case]['samples']) > lb, np.array(data[case]['samples']) < ub)) < target:
                proposal = np.random.uniform(lb, ub)
                acceptance = C.frb_distribution(proposal, case=case) / maxdNdz
                if acceptance > np.random.uniform():
                    data[case]['samples'].append(proposal)
    # Compute mean IGM dispersion measures for models
    print('Finished sampling of z values, computing DM values now')
    for modelname in models.keys():
        model = models[modelname]
        # Computing DM(z) involves integrating up to z, since we do this for many z
        # we can skip many integrations by making it one call (with large accuracy)
        # and them simply interpolating.
        zinterp = np.linspace(0, 30, 1000000)
        fast_DM_of_z = sip.interp1d(zinterp, C.toVal(
            model.fast_DM_array(zinterp, xe_func=model.xefunc_of_xifunc(xi_phys), rtol=1e-7, atol=1e-10),
            aun.pc / aun.cm ** 3))
        for case in phases.keys():
            filename = 'data/' + modelname + "_" + data_samples + "_" + case + '_norm' + str(number_total) + '.npy'
            DM_points = []
            for z in data[case]['samples']:
                DM_points.append([z, fast_DM_of_z(z)])
            DM_points = np.array(DM_points)
            print("Saving to ", filename)
            np.save(filename, DM_points)
        print('Done sampling DMs for ' + modelname + ' model.')


# Demonstrate distribution with a plot
zplot = np.linspace(5, 15, 1000)
number_total = 1000
plt.figure(figsize=(5,4))
plot_samples = 'samplesA'
for modelname in models.keys():
    for case in phases.keys():
        filename = 'data/' + modelname + "_" + plot_samples + "_" + case + '_norm' + str(number_total) + '.npy'
        DM_points = np.load(filename)
        Nbins = 21
        counts, bins, _ = plt.hist(DM_points.T[0], bins=Nbins, color=plt.cm.tab20b.colors[8], alpha=0.6, range=[5,15],
            label="Dataset of "+str(number_total)+" samples")
        counts, bins, _ = plt.hist(DM_points.T[0], bins=Nbins, color=plt.cm.tab20b.colors[8], alpha=1, range=[5,15],
            histtype="step")
        filename100 = 'data/' + modelname + "_" + plot_samples + "_" + case + '_norm' + str(100) + '.npy'
        plt.hist(np.load(filename100).T[0], bins=Nbins, color=plt.cm.tab20b.colors[12], alpha=1, range=[5,15],
            label="Dataset of "+str(100)+" samples")
        plt.plot(zplot, np.diff(bins)[0]*len(DM_points.T[0])*C.frb_distribution(zplot, case=case),
            color=plt.cm.tab20b.colors[0], alpha=0.75,
            label="Redshift distribution")
        plt.yscale("log")
        plt.ylim(0.5, 200)
        for i in range(10):
            plt.axhline(1+i, ls=':', alpha=0.5, color="black", lw=0.5)
        plt.xlim(5, 15)

handles, labels = plt.gca().get_legend_handles_labels()

plt.ylabel(r'FRB source distribution $\mathrm{d}N/\mathrm{d}z$')
plt.xlabel('Redshift $z$')
plt.tight_layout()
plt.legend(np.array(handles)[[2,0,1]], np.array(labels)[[2,0,1]])
plt.tight_layout()
plt.savefig("paper_plots/Fig1_A_frb_distribution_"+plot_samples+".pdf")
plt.savefig("paper_plots/Fig1_A_frb_distribution_"+plot_samples+".png", dpi=600)
plt.show()
