import numpy as np
from astropy.constants import c, h, k_B
k_B = k_B.si.value
h = h.si.value
c = c.si.value
T_CMB = 2.7255

def hx(freq):
	x = (h*freq)/(k_B*T_CMB)
	h_x = (x**4)* (np.exp(x)/(np.exp(x)-1)**2.)
	return h_x

def spec_cmb(freq):  
	intensity_0 = 2*k_B**3.*T_CMB**3./h**2./c**2. #The CMB spectrum is computed as $10^{20}\frac{I_0}{T_\mathrm{CMB}} h(x)$
	a = intensity_0 * hx(freq)
	intensity_cmb = a * 1e20 /T_CMB
	return(intensity_cmb)

def spec_ksz(freq):
	intensity_0 = 2*k_B**3.*T_CMB**3./h**2./c**2.
	intensity_ksz = intensity_0 * hx(freq)* 1e20
	return(intensity_ksz)


def fx(freq):
	x = (h*freq)/(k_B*T_CMB)
	f_x = (x*(np.exp(x)+1)/(np.exp(x)-1))-4
	return f_x

def spec_tsz(freq):
	intensity_0 = 2*k_B**3.*T_CMB**3./h**2./c**2.
	intensity_tsz = intensity_0 * hx(freq) *fx(freq)* 1e20
	return(intensity_tsz)
