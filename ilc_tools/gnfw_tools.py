import numpy as np
import astropy.units as u
from astropy.constants import pc, e,c,h,m_e, M_sun, sigma_T
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck15
from astropy.io import fits
from astropy.io import ascii
from scipy import integrate
from ilc_tools import sz_freq
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import ndimage

pc = pc.si.value
e = e.si.value
c = c.si.value
h = h.si.value
m_e = m_e.si.value
M_sun = M_sun.si.value
sigma_T = sigma_T.si.value
T_CMB = 2.7255

cosmo = FlatLambdaCDM(H0= 70, Om0 = 0.3)
H0 = cosmo.H0.value

def m500_2_r500(z, M_500):
	'''Conversion of M_500 to R_500
	
	Parameters
	----------
		z : float
		   1D array
		M_500: float
			1D array

	Returns
	--------
		R_500 in metres
	'''
	
	rho_crit = cosmo.critical_density(z).si.value
	R_500 = (M_500*M_sun*3/(4*np.pi*(500*rho_crit)))**(1./3) #in m
	
	return R_500
    
def r500_2_m500(r, z, factor = 500):
	'''Computes the Mass M_500 of a galaxy cluster from its radius r_500.
	M_500 is defined as the mass of a sphere with an average density 
	of 500 times the critical density and radius r_500. 
	Parameters
	----------
	r: float
		Cluster radius r_500
	z: float
		Cluster redshift
	factor: float, optional
		Overdensity factor. Default: 500
	Returns
	-------
	M_500: float
		Cluster Mass M_500
	'''

	rho_c = cosmo.critical_density(z).si.value * (1e6 * pc)**3 / M_sun
	M_500 = factor * rho_c * 4/3. * np.pi * r**3.
	return(M_500)
	

def gnfw(r, z, M_500, p='Arnaud' , xx = None, yy = None):
	'''Computes the radial 3D electron pressure profile of 
	a galaxy cluster using a GNFW model (Nagai et al. 2007)
	
	Parameters
	----------
		r: float
			Radial distance from the cluster center in meters
		z : float
			cluster redshift
		M_500: float
			cluster mass computed around R500
		p: string
			pressure profile to be used. Defualt: Arnaud
		xx: float
			min value of r along fixed axis. Used for projection only. Default : None
		yy: float
			max value of r along fixed axis. Used for projection only. Default: none

	Returns
	--------
		P_r : float
			Radial 3D electron pressure at radius r in units of J/m^3
	  
	'''
	
	if r is None:
		r = np.sqrt(xx**2 + yy**2)
		
	if p == 'Arnaud':
		P_0, c_500, gamma, alpha, beta = 8.403 * (H0/70.)**(-3/2.), 1.177,0.3081, 1.0510, 5.4905
	if p == 'Planck':
		P_0, c_500, gamma, alpha, beta = 6.41* (H0/70.)**(-3/2.), 1.81, 0.31, 1.33, 4.13
		
	h_70 = H0/70  
	R_500 = m500_2_r500(z, M_500)
	x = r/R_500
	p_x = P_0 / ((c_500 * x) ** gamma * (1 + ((c_500 * x )** alpha)) ** ((beta - gamma) / alpha))
	a = 0.12
	a_prime = 0.10 - ((a + 0.10)*(x/0.5)**3)/((1. + (x/0.5)**3))
	P_r = 1.65e-3 * cosmo.efunc(z)**(8/3) * (M_500/(3* 1e14 * (1/h_70)))**(2/3 + a + a_prime) * p_x * h_70**2 * e*1e9  #converting from keV/cm3 to J/m3 
	
	return P_r

def gnfw_projected(x1, z, M_500, p = 'Arnaud', r_max = 5.0, r_min = 1e-3, bins = 1000):
	'''Computes the radial Compton-y profile of a galaxy cluster 
	by numerically projecting a GNFW electron pressure model.
	(Nagai et al. 2007)
	
	Parameters
	-----------
		x1: float
			1D array. x axis fixed(y1 parallel to y- bounded by circular R_500) and x1 varied along y from origin.
		r: float
			1D array
			Default: None
		z : float
			cluster redshift
		M_500: float
			cluster mass computed around R500
		p: string
			GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Defualt: Arnaud
		r_min (xx): float
			Lower bound of the cluster radius in units of r_500. 
		A lower bound for the projection is necessary due to 
		the central cusp of the GNFW model. Default : 1e-3
		r_max (yy): float
			Cluster radius in units of r_500. Determines the line 
		of sight extent used during the projection. Default: 5.0
		bins: int
			Number of bins used for the projection along the line 
		of sight. Default: 1000
		
	Returns
	--------
		Compton_y : float
			Unitless radial Compton-y profile.
	
	'''
	
	n = len(x1)
	Compton_y = np.zeros(n)
	R_500 = m500_2_r500(z, M_500)

	for i in np.arange(n):
		if x1[i] > r_max*R_500:
			Compton_y[i] = 0
		else:
			if x1[i] < r_min*R_500:   #avoiding singularity
				x2_min = r_min*R_500
			else:
				x2_min = 0
			x2_max = np.sqrt((r_max*R_500)**2 - x1[i]**2)
			x2 = np.linspace(x2_min, x2_max, bins)
			P_r = gnfw(None, z, M_500, p, xx=x1[i], yy=x2)
			Compton_y[i] = (2* sigma_T)/(m_e *c**2) * integrate.simps(P_r,x2)  
				
	return Compton_y
	

def theta_compute(z, M_500):
	''' Calculate the projected angle.
	
	Parameters
	-----------
		z : float or float array
			cluster redshift
			
		M_500: float or float array
			 cluster mass computed within R500
	
	Returns
	-------------
		theta : float or float array
			Projected sky angle in units of arcmin
	'''
	
	R_500 = m500_2_r500(z, M_500)
	theta_500 = R_500/ (cosmo.angular_diameter_distance(z).si.value) * 180 / np.pi *60 #convert from m to radians to arcmin
	theta =np.linspace(0, 10*theta_500, 1000)
	
	return theta

def y_theta(z, M_500):
	'''Calculate the compton y profile.
	
	Parameters
	-----------
		z : float
			1D array redshift
			
		M_500: float
			1D array. mass computed around R500
	
	Returns
	--------
		y_profile : float or float array
			Compton y parameter for an array of projected angles
	'''
	
	theta = theta_compute(z, M_500)
	distance = (theta/ 60 *np.pi/180 )* (cosmo.angular_diameter_distance(z).si.value) #converting back to radians and then meters ie physical distance
	y_profile = gnfw_projected(distance, z, M_500, p = 'Arnaud', r_max = 5.0, r_min = 1e-3, bins = 1000)
	
	return y_profile
	
def T_e(r, z, M_500, xx = None, yy = None):
	'''Computes the radial 3D electron temperatue profile of a galaxy 
	cluster using the model presented by Vikhlinin et al. (2006). 
	The required value of the X-ray spectroscopic temperature T_x 
	is computed from the cluster mass M_500 using the M-T scaling 
	relation given by Reichert et al. (2011).No cool core assumed choosing an infinitesimally 
	small cooling radius. 
	Parameters
	----------
	r: float or float array
		Radial distance from the cluster center in meters
	z: float
		Cluster redshift
	M_500: float
		Mass of the cluster enclosed within r_500
	xx: float or float array
		x-coordinate. Used for projection only. Default: None
	yy: float or float array
		y-coordinate. Used for projection only. Default: None
	
	Returns
	-------
	T: float or float array
		Radial 3D electron temperatue profile of a galaxy 
		cluster in units of keV.
	'''
	
	if r is None:
		r = np.sqrt(xx**2 + yy**2)
		
	R_500 = m500_2_r500(z, M_500)
	r_cool = 1e-8*R_500
	E = cosmo.H(z).value/cosmo.H0.value
	T_x = ((M_500/1e14)/0.291 * (1/(E**(-1.04))))**(1/1.62)
	T = 1.35 * T_x * ((r/r_cool)**1.9 + 0.45) / ((r/r_cool)**1.9 + 1) / (1+ (r/(0.6*R_500))**2.)**0.45 
	
	return (T)
	
def tau_compute(r, z, M_500, p = "Arnaud", r_max = 5, r_min = 1e-3, bins = 1000):
	'''Computes the radial 2D integrated profile of the optical depth 
	of a galaxy cluster using the 3D pressure model given by 
	Arnaud et al. (2010) and the 3D tempeature modele presented by 
	Vikhlinin et al. (2006). The required value of the X-ray 
	spectroscopic temperature T_x is computed from the cluster 
	mass M_500 using the M-T scaling relation given by Reichert 
	et al. (2011). An infinitesimally small cooling radius assumed.
	
	Parameters
	----------
	r: float or float array
		Radial distance from the cluster center in meters
	z: float
		Cluster redshift
	M_500: float
		Mass of the cluster enclosed within r_500
	p: float array or string, optional
		GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Default: 'Arnaud'
	r_max: float, optional
		Cluster radius in units of r_500. Determines the line 
		of sight extent used during the projection. Default: 5.0
	r_min: float, optional
		Lower bound of the cluster radius in units of r_500. 
		A lower bound for the projection is necessary due to 
		the central cusp of the GNFW model. Default: 1e-3
	bins: float, optional
		Number of bins used for the projection along the line 
		of sight. Default: 1000

	Returns
	-------
	tau: float or float array
		2D radial optical depth profile of a galaxy cluster.
	'''
	
	R_500 = m500_2_r500(z, M_500)
	y = r / R_500

	tau = []

	for yy in y:
		if yy <= r_max:
			if yy < r_min: 
				x_min = r_min 
			else: 
				x_min = 0 
			x = np.linspace(x_min,np.sqrt(r_max**2-yy**2),bins)
			r = np.sqrt(yy**2. + x**2.) * R_500
			integrant = gnfw(r, z, M_500, p)/(T_e(r, z, M_500)*1000*e) # 1000*e is 1.6e-16 ie 1keV to J
			tau.append(2* sigma_T* integrate.simps(integrant, x*R_500))
		else:
			tau.append(0)

	return (np.array(tau))
	
def tau_theta(z, M_500):
	''' Compute the tau profile for a given sky angle.
	Parameters:
	-----------
	z: float
		cluster redshift
	M_500: float
		cluster mass within R_500
		
	Returns
	-------
	tau_profile: float array
		tau_profile for projected sky angles
	'''
	
	
	theta = theta_compute(z, M_500)
	distance = (theta/ 60 *np.pi/180 )* (cosmo.angular_diameter_distance(z).si.value) #converting back to radians and then meters ie physical distance
	tau_profile = tau_compute(distance, z, M_500, p = 'Arnaud', r_max = 5.0, r_min = 1e-3, bins = 1000)
	
	return tau_profile

def simulate_cluster(z, M_500, p="Arnaud", r_max = 5.0, r_min = 1e-3, npix = 400, pxsize = 1.5, bins=1000, dx = 0, dy = 0):
	'''Computes Compton-y and optical depth maps of a galaxy cluster 
	at with mass M_500 at redshift z by numerically projecting and 
	weighting the pressure model given by Arnaud et al. (2010) and the 
	electron temperature model presented by Vikhlinin et al. 
	(2006). The required value of the X-ray spectroscopic 
	temperature T_x is computed from the cluster mass M_500 
	using the M-T scaling relation given by Reichert et al. 
	(2011). Cool core is not assumed.
	Parameters
	----------
	z: float
		Cluster redshift
	M_500: float
		Mass of the cluster enclosed within r_500
	p: float array or string, optional
		GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Default: 'Arnaud'
	r_min: float, optional
		Lower bound of the cluster radius in units of r_500. 
		A lower bound for the projection is necessary due to 
		the central cusp of the GNFW model. Default: 1e-3
	r_max: float, optional
		Cluster radius in units of r_500. Determines the line 
		of sight extent used during the projection. Default: 5.0
	map_size: int, optional
		Size of the map in degrees. Default: 10
	pixel_size: float, optional
		Pixel size in arcmin. Default: 1.5
	bins: float, optional
		Number of bins used for the projection along the line 
		of sight. Default: 1000
    dx: float, optional
		Offset of cluster center from image center along 
		x-axis in pixels. Default: 0
	dy: float, optional
		Offset of cluster center from image center along 
		y-axis in pixels. Default: 0
	Returns
	-------
	cluster_map: 2D float array
		unitless simulated y-map and tau-map
	'''
	
	
	YY, XX = np.mgrid[0:npix,0:npix]
	center = (npix/2+dx,npix/2+dy)
	theta = np.radians(np.sqrt((XX-center[0])**2 + (YY-center[1])**2)*pxsize/60)
	distance = theta*cosmo.angular_diameter_distance(z).si.value
	
	#distance = distance.reshape(npix*npix)
	
	r_interp = np.linspace(np.min(distance), np.max(distance),bins)
	
	y_interp = gnfw_projected(r_interp, z, M_500, p, r_max=r_max, r_min=r_min, bins=bins)
	tau_interp = tau_compute(r_interp, z, M_500, p , r_max = r_max, r_min = r_min, bins = bins)
	
	y_map = np.interp(distance, r_interp, y_interp)
	tau_map = np.interp(distance, r_interp, tau_interp)
	
	cluster_map = np.array([y_map, tau_map])
	
	return(cluster_map)

def simulate_tsz_sky(z, M_500, npix, pxsize, freq, fwhm,bins = 1000, dx = 0, dy = 0, p = "Arnaud", MJy = True, convolve = True):
    
    cluster_map = simulate_cluster(z, M_500, p, r_max = 5.0, r_min = 1e-3, npix = npix, pxsize =pxsize , bins= bins, dx = dx, dy = dy)
    y_map = cluster_map[0]
    
    intensity_tsz = sz_freq.spec_tsz(freq)
    frequency = sz_freq.fx(freq)
    print(frequency, intensity_tsz)
    
    nf = len(freq)
    
    tsz_maps = np.zeros((nf, npix, npix))
    if MJy is True:
        for i in np.arange(nf):
            tsz_maps[i,:,:] = intensity_tsz[i] * y_map 
    else:
        for i in np.arange(nf):
            tsz_maps[i,:,:] = frequency[i] * y_map * T_CMB
            
    if convolve is True:
        sigma_beam = fwhm / (2*np.sqrt(2*np.log(2))) / pxsize
        for i in np.arange(nf):
            tsz_maps[i,:,:] = ndimage.gaussian_filter(tsz_maps[i], sigma=sigma_beam, order=0, mode='wrap', truncate=20.0)
        
    return (tsz_maps, y_map)
    
def simulate_ksz_sky(z, M_500,npix, pxsize, freq, fwhm, v_pec, bins = 1000, dx = 0, dy = 0, p = "Arnaud", MJy = True, convolve = True): #v_pec in km/s
    
    
    cluster_map = simulate_cluster(z, M_500, p, r_max = 5.0, r_min = 1e-3, npix = npix, pxsize =pxsize , bins= bins, dx = dx, dy = dy)
    tau_map = cluster_map[1]
    print(tau_map.max())
    print(c)
    
    nf = len(freq)
    sigma_beam = fwhm / (2*np.sqrt(2*np.log(2))) / pxsize
    intensity_ksz = sz_freq.spec_ksz(freq)
    print(intensity_ksz)
    
    ksz_maps = np.zeros((nf,npix,npix))        
    if MJy is True:
        for i in np.arange(nf):
            ksz_maps[i,:,:] = intensity_ksz[i] * tau_map * (v_pec *1e3 /c)
    else:
        for i in np.arange(nf):
            ksz_maps[i, :, :] = tau_map * (v_pec *1e3 /c) * T_CMB
        
    if convolve is True:
        for i in np.arange(nf):
            ksz_maps[i,:,:] = ndimage.gaussian_filter(ksz_maps[i], sigma=sigma_beam, order=0, mode='wrap', truncate=20.0)
        
    return (ksz_maps, tau_map) 
