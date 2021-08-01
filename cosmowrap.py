import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as smi
import scipy.stats as sst
import scipy.special as ssp
import scipy.optimize as sop
import scipy.integrate as sin
import scipy.constants as sco
import scipy.interpolate as sip
import astropy.units as aun
import astropy.constants as aco
from copy import deepcopy

# Just a few wrapper functions around astropy, FRBs etc.
# I plan to include more things, e.g. wrappers for CLASS functions are in progress.

# A few global variables because they might be used by the user as arguments
planck18_class_bestfit_ncdm = {'A_s': 2.1005e-9,'n_s': 0.96605, 'tau_reio': 0.0543, 'omega_b': 0.02238,'omega_cdm': 0.1201, 'h': 0.6732, 'N_ncdm': 1, 'N_ur': 2.03066, 'omega_ncdm': 0.000645}
planck_lcdm_bestfit_class = {'A_s': 2.105e-9,'n_s': 0.9665, 'tau_reio': 0.0561, 'omega_b': 0.02242,'omega_cdm': 0.11933, 'h': 0.6766}
planck15_class_bestfit_ncdm = {'YHe': 0.245352, 'A_s': 2.140509e-09,'n_s': 0.9682903, 'tau_reio': 0.06664549, 'omega_b': 0.02227716,'omega_cdm': 0.1184293, 'h': 0.6786682, 'N_ncdm': 1, 'N_ur': 2.03066666667, 'omega_ncdm': 0.0006451439}
planck15_lcdm_bestfit_cosmomc = {'logA': 3.094,'ns': 0.9645, 'tau': 0.079, 'omegabh2': 0.02225,'omegach2': 0.1198, 'theta': 1.04077}
planck_lcdm_bestfit_cosmomc = {'logA': 3.047,'ns': 0.9665, 'tau': 0.0561, 'omegabh2': 0.02242,'omegach2': 0.11933, 'theta': 1.04101}
planck_lcdm_priors_cosmomc = {"omegabh2": {"prior": {"dist": "norm", "loc": 0.02242, "scale": 0.00014},"latex": r"\Omega_\mathrm{b} h^2"}, "omegach2": {"prior": {"dist": "norm", "loc": 0.11933, "scale": 0.00091},"latex": r"\Omega_\mathrm{c} h^2"}, "theta": {"prior": {"dist": "norm", "loc": 1.04101, "scale": 0.0029},"latex": r"100\theta_\mathrm{MC}"}, "tau": {"prior": {"dist": "norm", "loc": 0.0561, "scale": 0.0071},"latex": r"\tau"}, "logA": {"prior": {"dist": "norm", "loc": 3.047, "scale": 0.014},"latex": r"\log(10^{10} A_\mathrm{s})"}, "ns": {"prior": {"dist": "norm", "loc": 0.9665, "scale": 0.0038},"latex": r"n_\mathrm{s}"}}
planck_lcdm_refs_cosmomc = {"omegabh2": {"ref": {"dist": "norm", "loc": 0.02242, "scale": 0.00014},"latex": r"\Omega_\mathrm{b} h^2"}, "omegach2": {"ref": {"dist": "norm", "loc": 0.11933, "scale": 0.00091},"latex": r"\Omega_\mathrm{c} h^2"}, "theta": {"ref": {"dist": "norm", "loc": 1.04101, "scale": 0.0029},"latex": r"100\theta_\mathrm{MC}"}, "tau": {"ref": {"dist": "norm", "loc": 0.0561, "scale": 0.0071},"latex": r"\tau"}, "logA": {"ref": {"dist": "norm", "loc": 3.047, "scale": 0.014},"latex": r"\log(10^{10} A_\mathrm{s})"}, "ns": {"ref": {"dist": "norm", "loc": 0.9665, "scale": 0.0038},"latex": r"n_\mathrm{s}"}}
wmap_partial_posteriors_cosmomc = {"omegabh2": {"prior": {"dist": "norm", "loc": 0.02213, "scale": 0.00061},"latex": r"\Omega_\mathrm{b} h^2"}, "omegach2": {"prior": {"dist": "norm", "loc": 0.1136, "scale": 0.0056},"latex": r"\Omega_\mathrm{c} h^2"}, "H0": {"prior": {"dist": "norm", "loc": 64.5, "scale": 4.4},"latex": r"H_0"}, "tau": {"prior": {"dist": "norm", "loc": 0.086, "scale": 0.014},"latex": r"\tau"}}
planck_neutrinos_bestfit_class = {'N_ncdm': 1, 'N_ur': 2.0328, 'm_ncdm': 0.06}
class_max_ranges = {"omega_b":{"prior":{"dist":"uniform","min":5e-3,"max":3.9e-2}},"omega_cdm":{"prior":{"dist":"uniform","min":0,"max":1}},"H0":{"prior":{"dist":"uniform","min":30,"max":100}},"tau_reio":{"prior":{"dist":"uniform","min":4e-3,"max":0.2}}}

def dict_to_CLASS(d):
	for key, val in d.items():
		print(key, '=', val)

def class_verbose(i):
		return {'input_verbose': i, 'background_verbose': i, 'thermodynamics_verbose': i, 'perturbations_verbose': i, 'transfer_verbose': i, 'primordial_verbose': i, 'spectra_verbose': i, 'nonlinear_verbose': i, 'lensing_verbose': i, 'output_verbose': i}

def CLASS_to_astropy(d):
	astrokwargs = {}
	astrokwargs['H0'] = d['h']*100
	astrokwargs['Om0'] = (d['omega_b']+d['omega_cdm'])/d['h']**2
	astrokwargs['Ob0'] = d['omega_b']/d['h']**2
	if 'mnu' in d:
		astrokwargs['m_nu'] = d['m_nu']
	return astrokwargs

# The basic class containing all the simple functions, should be basically instant
class _base_():
	def __init__(self, params):
		if 'verbose' in params:
			print('Initializing _base_')

		self.clikpath = params.pop("clikpath")
		self.datapath = params.pop("datapath")
		self.m_H = 1.0078250322*aun.u #https://en.wikipedia.org/wiki/Isotopes_of_hydrogen
		self.m_He = 4.0026032541*aun.u #https://en.wikipedia.org/wiki/Isotopes_of_helium
		# A collection of variables that are =1 in natural units (Planck units)
		# Note that h != 1 because hbar=1, and 1eV energy != 1eV wavelength!
		self.hbar = aco.hbar
		self.c = aco.c
		self.k_B = aco.k_B
		self.Energy_to_Temperature = 1/self.k_B
		self.Energy_to_iLength = 1/self.c/self.hbar
		self.Energy_to_Mass = 1/self.c**2
		self.Energy_to_iTime = 1/self.hbar
		self.Energy_to_SIDensity = self.Energy_to_iLength**3/self.c**2
		
		# Trivial cosmological functions, conversions and others		
		self.z_of_a = lambda a: 1/a-1
		self.a_of_z = lambda z: 1/(z+1)
		self.lna_of_a = lambda a: np.log(a)
		self.a_of_lna = lambda lna: np.exp(lna)
		self.z_of_lna = lambda lna: self.z_of_a(self.a_of_lna(lna))
		self.lna_of_z = lambda z: self.lna_of_a(self.a_of_z(z))
		self.dzdlna_of_z = lambda z: -1/self.a_of_z(z)
		self.xHI_of_xi = lambda x: 1-x
		self.xi_of_xHI = lambda x: 1-x
		self.CV_tt = lambda l, Dl_tt: np.sqrt(2/(2*l+1))*Dl_tt
		self.CV_te = lambda l, Dl_tt, Dl_te, Dl_ee: np.sqrt(2/(2*l+1))*np.sqrt(Dl_tt*Dl_ee+Dl_te**2)
		self.CV_ee = lambda l, Dl_ee: np.sqrt(2/(2*l+1))*Dl_ee

	def mean(self, x, w=None):
		if np.any(w==None):
			print('No weights given, assuming ones.')
			w=np.ones(len(x))
		return np.average(x, weights=w)
	def std(self, x, w=None, ddof=1):
		if np.any(w==None):
			print('No weights given, assuming ones.')
			w=np.ones(len(x))
		N = len(x)
		# From https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
		var = N/(N-1)*np.average((x-np.average(x, weights=w))**2, weights=w)
		return np.sqrt(var)

	def toVal(self, x, xunit):
		return (x*aun.one).to(xunit).value
	
	def checkUnit(self, x, xunit=aun.one):
		self.toVal(x, xunit)
	
	# Unit functions:
	#   decompose() to simplify units
	#   to() to convert to certain units
	def interp1d(self, x, y, xunit=aun.one, yunit=aun.one, kind='cubic', **kwargs):
		# Unit-aware (cubic) interp1d wrapper (cubic spline)
		x = self.toVal(x, xunit)
		y = self.toVal(y, yunit)
		interpolation = sip.interp1d(x,y, kind=kind, **kwargs) 
		return lambda z: interpolation(self.toVal(z, xunit))*yunit 
	
	def bisect(self, f, a, b, xunit=aun.one, yunit=aun.one, **kwargs):
		# Unit-aware bisect wrapper
		a = self.toVal(a, xunit)
		b = self.toVal(b, xunit)
		return sop.bisect(lambda x: self.toVal(f(x*xunit), yunit), a, b, **kwargs)*xunit
	
	def derivative(self, f, x0, dx, xmin=None, xmax=None, xunit=aun.one, yunit=aun.one, **kwargs):
		# Unit-aware derivative wrapper with mandatory dx and xlims
		if xmin==None:
			xmin=-np.inf
		else:
			xmax = self.toVal(xmax, xunit)
		if xmax==None:
			xmax=np.inf
		else:
			xmin = self.toVal(xmin, xunit)
		dx = self.toVal(dx, xunit)
		x0 = self.toVal(x0, xunit)
		assert(xmax>xmin)
		assert(xmax-dx>xmin+dx)
		if x0+dx>xmax:
			return (f(x0*xunit, **kwargs)-f((x0-dx)*xunit, **kwargs))/(dx*xunit)
		elif x0-dx<xmin:
			return (f((x0+dx)*xunit, **kwargs)-f(x0*xunit, **kwargs))/(dx*xunit)
		else:
			return yunit*smi.derivative(lambda x: self.toVal(f(x*xunit, **kwargs), yunit), x0, dx=dx)/xunit
	
	def tanh_step(self, z, z0, dz=0.5):
		# Reionization step function used e.g. in Planck analysis
		# Common values for z0 are 7.8 for H and He (first) and
		# 3.5 for second He ionization; dz=0.5 is usually assumed.
		y=(1+z)**(3/2)
		y0=(1+z0)**(3/2)
		dy=3/2.*(1+z0)**(1/2)*dz
		return (1+np.tanh((y0-y)/dy))/2

	def planck_globaltau_loglike(self, global_tau_value, bestfit=0.0561, sigma=0.0071):
		# https://arxiv.org/abs/1807.06209
		# 2018: default; 2016: 0.058 +/- 0.012
		return sst.norm.logpdf(global_tau_value, loc=bestfit, scale=sigma)
	def planck_earlytau_loglike(self, early_tau_value, limit=0.019/2):
		# From Heinrich et al. 2021 (https://arxiv.org/abs/2104.13998),
		# but just approximating this as normal distribution tail.
		# Also note that 0.019 is for (15,50) and 0.020 for (15,30)
		return sst.norm.logpdf(early_tau_value, loc=0, scale=limit)

class _data_(_base_):
	def __init__(self, params):
		super().__init__(params)
		if 'verbose' in params:
			print('Initializing _data_')
		self.data_extended = False

		# From Behroozi et al, https://arxiv.org/abs/1806.07893
		self.behroozi_csfrs = np.genfromtxt("Behroozi2019_csfrs.dat").T
		self.behroozi_a = self.behroozi_csfrs[0]
		self.behroozi_z = self.z_of_a(self.behroozi_a)
		self.behroozi_log10CSFRD = np.log10(self.behroozi_csfrs[7])
		self.log10cCSFRD_Behroozi_2019 = self.interp1d(self.behroozi_z, self.behroozi_log10CSFRD, kind='linear', fill_value='extrapolate')
		self.cCSFRD_Behroozi_2019 = lambda z: 10**self.log10cCSFRD_Behroozi_2019(z)*aun.M_sun/aun.yr/aun.Mpc**3

	def frb_distribution(self, z, case=None, prefactor=None, zmin=5, zmax=15):
		if prefactor == None:
			if not self.data_extended:
				self.extend_data(zmin=zmin, zmax=zmax)
			else:
				assert self.zmin_frbdist == zmin
				assert self.zmax_frbdist == zmax
			prefactor = self.prefactors[case]

		if case=='uni':
			return prefactor*np.heaviside(self.zmax_frbdist-z,0)*np.heaviside(z-self.zmin_frbdist,0)
		elif case=='sfr':
			return prefactor*self.CSFR_Msun_per_year(z)*np.heaviside(self.zmax_frbdist-z,0)*np.heaviside(z-self.zmin_frbdist,0)
		elif case=='exp':
			return prefactor*z*np.exp(-z)*np.heaviside(self.zmax_frbdist-z,0)*np.heaviside(z-self.zmin_frbdist,0)
		else:
			assert False, "Case "+str(case)+" not found."

	def extend_data(self, zmin=5, zmax=15):
		self.zmin_frbdist = zmin
		self.zmax_frbdist = zmax
		# Copy CSFR into interpolation
		CSFRarr = []
		dz = 0.01
		zarr = np.arange(0,21,dz)
		for z in zarr:
			R0 = self.comov_dist(z)
			R1 = self.comov_dist(z+dz)
			dR = R1-R0
			dV = 4*np.pi*R0**2*dR
			CSFRarr.append(self.toVal(dV*self.cCSFRD_Behroozi_2019(z), aun.M_sun/aun.yr))
		CSFRarr = np.array(CSFRarr)
		self.CSFR_Msun_per_year = self.interp1d(zarr, CSFRarr, fill_value=0, bounds_error=False)
		# Normalization
		cases = ['uni', 'exp', 'sfr']
		self.prefactors = {'uni': 0, 'exp': 0, 'sfr': 0}
		for case in cases:
			f = lambda z: self.frb_distribution(z, case=case, prefactor=1)
			total = sin.quad(f,self.zmin_frbdist, self.zmax_frbdist,  epsrel=1e-4, epsabs=0.01)[0]
			self.prefactors[case] = 1/total
		self.data_extended = True

class _frb_(_base_):
	# I haven't been very accurate with units in this part
	def __init__(self, params):
		super().__init__(params)
		if 'verbose' in params:
			print('Initializing _frb_')
		from astropy.cosmology import FlatLambdaCDM
		astrokwargs = CLASS_to_astropy(params)
		cosmo = FlatLambdaCDM(**astrokwargs)
		self.YHe = 0.2454 # From PArthENoPE, mentioned in page 4 of 1807.06209
		self.He_to_H_number_ratio = self.YHe/(1-self.YHe)*self.m_H/self.m_He
		self.xe1_of_xi = lambda xi: xi*(1+self.He_to_H_number_ratio)
		self.H = cosmo.H
		self.n_H0 = cosmo.critical_density0*cosmo.Ob0*(1-self.YHe)/self.m_H
		self.n_H = lambda z: self.n_H0*(1+z)**3
		self.comov_dist = cosmo.comoving_distance
		self.lumi_dist = cosmo.luminosity_distance

	def xe_of_xi(self, z, zreio1=7.82, zreio2=3.5, dz=0.5):
		return self.xe1_of_xi(self.tanh_step(z, zreio1, dz=dz))+self.He_to_H_number_ratio*self.tanh_step(z, zreio2, dz=dz)

	def xefunc_of_xifunc(self, xi_func, zreio2=3.5, dz2=0.5):
		return lambda z: self.xe1_of_xi(xi_func(z))+self.He_to_H_number_ratio*self.tanh_step(z, zreio2, dz=dz2)

	def xifunc_Planck(self, zreio1=7.82, dz=0.5):
		return lambda z: self.tanh_step(z, zreio1, dz=dz)

	def optical_depth_of_xe(self, xe_func, zlow=0, zhigh=50, epsrel=1e-5, epsabs=0, debug=False, integrate_mode=None):
		prefactor = self.n_H(0)*aco.c*aco.sigma_T*aun.Gyr
		f = lambda z: xe_func(z)*(1+z)**2/self.toVal(self.H(z), aun.Gyr**-1)
		if integrate_mode is None:
			r, e = sin.quad(f, zlow, zhigh, epsrel=epsrel, epsabs=epsabs)
		else:
			zint = np.linspace(zlow, zhigh, int(integrate_mode))
			fint = f(zint)
			r = np.average(fint)*(zhigh-zlow)
			e = np.abs((np.average(fint[1:])-np.average(fint[:-1])))*(zhigh-zlow)
		if debug:
			print('[Debug] Integral = {0:.6f} +/- {1:.6f}'.format(r,e))
		if(e > r*epsrel*10):
			print('Warning: Low Optical Depth intergral accuracy of r, e =',r,e,'epsrel =',epsrel)
		assert r >= e/epsrel/1e3 or r+e<1e-8, "ERROR: Integral accuracy insufficient"
		return self.toVal(prefactor*r,aun.one)

	def optical_depth_of_xi(self, xi_func, **kwargs):
		xe_func = self.xefunc_of_xifunc(xi_func)
		return self.optical_depth_of_xe(xe_func, **kwargs)

	def fast_DM_array(self, zarr, xe_func=None, xi_func=None, rtol=1e-5, atol=1e-5):
		# Use solve_ivp to solve the integration in one go for for all z values
		zmin = np.min(zarr)
		zmax = np.max(zarr)
		assert (xe_func is None) != (xi_func is None), "Please pass pass xi_func XOR xe_func"
		assert zmin >= 0, "ERROR: Can't compute DM for z<0"
		xe_func = xe_func if xe_func is not None else self.xefunc_of_xifunc(xi_func)
		f = lambda z, y: self.toVal(self.n_H(0)*aco.c*xe_func(z)*(1+z)/self.H(z), aun.pc/aun.cm**3)
		res = sin.solve_ivp(f, [0, zmax] ,[0], t_eval=zarr, rtol=rtol/len(zarr), atol=atol/len(zarr))
		assert res['status']==0, 'ERROR: solve_ivp failed for some reason'
		return res['y'][0]*aun.pc/aun.cm**3

	def dDMdz(self, z, xe_func=None):
		return self.toVal(self.n_H(0)*aco.c*xe_func(z)*(1+z)/self.H(z), aun.pc/aun.cm**3)

	def sigmaDM_obs_of_z_and_DM(self, z, DM, alpha=-1.5, calibration_factor=1, tScatter=0*aun.ms):
		# Parameters for FAST-like telescope:
		nu = 1400*aun.MHz
		deltanu = 400*aun.MHz
		Tsys = 35*aun.K
		nchan = 4096
		tsamp = 100*aun.us
		G = 15*aun.K/aun.Jy
		deltanuchan = deltanu/nchan
		# Definition of constant
		k = 4.15e6*aun.MHz**2*aun.ms*aun.cm**3/aun.pc # == 4.15*aun.GHz**2*aun.ms*aun.cm**3/aun.pc
		# Re-calculate luminosity distance and DM for redshift,
		dL = self.toVal(self.lumi_dist(z), aun.Mpc)
		Scalib = 12.3*aun.Jy*calibration_factor
		# use FRB180924 (Banniuster et al, https://arxiv.org/abs/1906.11476)
		dLcalib = self.toVal(self.lumi_dist(0.3214), aun.Mpc)
		S = Scalib*(dLcalib/dL)**2*((1+z)/(1+0.3214))**(1+alpha)
		Wint0 = 1*aun.ms
		Wint = Wint0*(1+z)
		prefactor = nu**3/k*Tsys/S/G
		tDM = k*DM*deltanuchan/nu**3
		W = np.sqrt(Wint**2+tDM**2+tScatter**2)
		root = np.sqrt(W/2/deltanu**3)
		sigmaDM = (prefactor*root).to(aun.pc/aun.cm**3)
		SN = S*G*np.sqrt(2*deltanu*W)/Tsys
		return sigmaDM

	def sigma_DM_IGM_of_z(self, z, **kwargs):
		zinterp = np.linspace(0.01,z,1000)
		sigmas = self.fast_DM_array(zinterp, **kwargs)*0.2/np.sqrt(zinterp)
		return np.max(sigmas)

	def sigma_DM_IGM_function(self, zmin=0.01, zmax=100, **kwargs):
		zinterp = np.linspace(zmin,zmax,1000)
		sigmas = self.fast_DM_array(zinterp, **kwargs)*0.2/np.sqrt(zinterp)
		cumMax = np.maximum.accumulate(sigmas)
		return self.interp1d(zinterp, cumMax, yunit=aun.pc/aun.cm**3)


def cosmology(modules=None, params=None, z_max_pk=1000,  P_k_max=1.3/aun.Mpc, l_max_scalars=2508, lensing='yes', datapath="/home/stefan/phd/codes/cosmowrap/data/", clikpath="/home/stefan/.links/planck_data/"):
	if params==None:
		params = planck_lcdm_bestfit_class
	if 'output' not in params:
		params['output'] = ''

	if modules==None:
		modules = 'base boltzmann data thermal'

	params['datapath'] = datapath
	params['clikpath'] = clikpath

	# Ordering of classes in python, for inheritance
	# Ordering: * Children must be before their ancestors:
	#             When X(A,B), A must be before B
	#           ==> Put the ones without children to the top
	inheritance = []
	if 'frb' in modules:
		inheritance.append(_frb_)
	if 'data' in modules:
		inheritance.append(_data_)
	if 'base' in modules:
		inheritance.append(_base_)
	assert inheritance != [], 'Missing modules, missspelled?'
	class combinedclass(*inheritance):
		def __init__(self):
			super().__init__(params)
	return combinedclass()
