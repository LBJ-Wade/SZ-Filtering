import numpy as np
import astropy.units as u
from astropy.constants import pc, e,c,h,m_e, M_sun, sigma_T, m_p
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck15
from scipy import integrate
from ilc_tools import gnfw_tools as gnfw

m_p = m_p.si.value
sigma_T = sigma_T.si.value

cosmo = FlatLambdaCDM(H0= 70, Om0 = 0.3)
c = c.si.value
T_CMB = 2.7255
#v_pec = 300

#constants
f_b = 0.175
f_gas = 0.125
mu_e = 1.14 #1.17

#AGN feedback 
rho_0 = 4e3
alpha = 0.88
beta = 3.83
x_core = 0.5
gamma = -0.2

#z = 0.08
#M_500 = 6.0e14 

#radius-mass conversion
def r200(z, M_500):
    R_500 = gnfw.m500_2_r500(z, M_500)
    R_200 = R_500/0.65 #for c = 4
    M_200 = M_500 * (200/500) * (0.65)**(-3)
    #print(R_200, R_500, M_200)
    return R_200

#electron density    
def electron_density(z, M_500, r, x_core, profile = 'battaglia',  xx = None, yy = None):
    if r is None:
        r = np.sqrt(xx**2 + yy**2)
    R_200 = r200(z, M_500)
    x = r/R_200 
    
    if profile == 'battaglia':
        alpha, beta, rho_0, gamma = 0.88, 3.83, 4e3, -0.2
    C = (beta + gamma)/alpha
    
    rho_fit = rho_0 * (x/x_core)**gamma * (1 + (x/x_core)**alpha)**(-C)
    
    rho_crit = cosmo.critical_density(z).si.value
    rho_gas = rho_crit* rho_fit* f_gas
    
    n_e = rho_gas/(mu_e * m_p)
    return n_e
    
#tau profile
def tau(z, M_500,r, x_core, profile = "battaglia", r_max = 5, r_min = 1e-3, bins = 1000):
    
    R_200 = r200(z, M_500)
    y = r / R_200
    tau = []

    for yy in y:
        if yy <= r_max:
            if yy < r_min: 
                x_min = r_min 
            else: 
                x_min = 0 
            x = np.linspace(x_min,np.sqrt(r_max**2- yy**2),bins)
            r = np.sqrt(yy**2. + x**2.) * R_200
            integrant = electron_density(z, M_500, r,x_core, profile = 'battaglia')
            tau.append(sigma_T* integrate.simps(integrant, x*R_200))
        else:
            tau.append(0)

    return (np.array(tau))
    
def theta_compute(z, M_500):
    R_200 = r200(z, M_500)
    theta_200 = R_200/ (cosmo.angular_diameter_distance(z).si.value) * 180 / np.pi *60 
    theta =np.linspace(0, 10*theta_200, 50) #bin_width = 10/50 = 0.2
    return theta, theta_200

def tau_theta(z, M_500):
    theta, theta_200 = theta_compute(z, M_500)
    distance = (theta/ 60 *np.pi/180 )* (cosmo.angular_diameter_distance(z).si.value)
    tau_profile = tau(z, M_500,distance, x_core, profile = 'battaglia', r_max = 5.0, r_min = 1e-3, bins = 1000)
    return tau_profile
    
def simulate_cluster(z, M_500, x_core, profile="battaglia", r_max = 5.0, r_min = 1e-3, map_size = 10, pxsize = 1.5, bins=1000):
    
    npix = int(map_size*60/pxsize)
    YY, XX = np.mgrid[0:npix,0:npix]
    center = (npix/2,npix/2)
    theta = np.radians(np.sqrt((XX-center[0])**2 + (YY-center[1])**2)*pxsize/60)
    distance = theta*cosmo.angular_diameter_distance(z).si.value
    r_interp = np.linspace(np.min(distance), np.max(distance), 1000)
    tau_interp = tau(z, M_500,r_interp, x_core, profile , r_max = r_max, r_min = r_min, bins = bins)
    tau_map = np.interp(distance, r_interp, tau_interp)
    
    return tau_map

