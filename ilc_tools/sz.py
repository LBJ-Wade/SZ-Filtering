h = 6.626e-27
c = 3.0e+10
k = 1.38e-16  #erg/K || k_b = 8.617e-5 #keV/K
y = 1e-4
T_CMB = 2.725
v = 1e+8     #peculiar velocity of cluster

def planck(frequency, T):
	'''Calculates the blackbody spectrum given frequency and temperature.
	
	Parameters
	-----------
		frequency: 1D array
			a range of integers in GHz
		T: integer
			in Kelvins
	
	Returns
	--------
		Intensity in MJy/sr
	'''
	
	a = 2.0*h*freq**3
	b = (h*freq)/(k*T)
	intensity = a/((c**2) * (np.exp(b)-1))
	
	return intensity* 1e23 /1e6 *0.001
	

def TSZ(frequency, T_CMB = 2.755): 
	''' Calculates the thermal sunyaev zel'dovich spectrum.
	
	Parameters
	----------
		frequency:  1D array
			ange of integers in GHz
		T_CMB: imported value of CMB temperature
			depends on the package imported Default : 2.755
	 
	 Returns
	 --------
		intensity in MJy/sr
	 '''
	 
	I_0 = 2*(k*T_CMB)**3/((h*c)**2)
	I= I_0 * 1e23 /1e6 #conversion to MJy/sr
	x = (h*frequency)/(k*T_CMB)
	f_x = (x*(np.exp(x)+1)/(np.exp(x)-1))-4
	h_x = (x**4)*np.exp(x)/(np.exp(x)-1)**2
	intensity_tSZ = y*I*f_x*h_x
	
	return intensity_tSZ
	
def kSZ(frequency, T_CMB):
	''' Calculates the kinetic sunyaev zel'dovich spectrum.
	
	Parameters
	----------
		frequency:  1D array
			range of integers in GHz
		T_CMB: imported value of CMB temperature
			depends on the package imported Default : 2.755
	 
	Returns
	--------
		intensity in MJy/sr
	'''
	 
	I_0 = 2*(k*T_CMB)**3/((h*c)**2)
	I= I_0 * 1e23 /1e6 #conversion to MJy/sr
	x = (h*frequency)/(k*T_CMB)
	h_x = (x**4)*np.exp(x)/(np.exp(x)-1)**2
	tau = y * (511/10) #optical depth_tau = y*m_e*c**2/(K_B*T)= K_B*T=10KeV and m_e*c**2= 511KeV
	intensity_kSZ = -tau*(v/c)*h_x*I
	
	return intensity_kSZ
	
def temp_tSZ(frequency):
	'''Calculates the temperature of tSZE effect.
	
	Parameters:
	-----------
		frequency: 1D array
			 range of intgers in GHz
			   
	 Returns:
	 ---------
		temperature in kelvin
	 '''
	
	x = (h*frequency)/(k*T_CMB)
	f_x = (x*(np.exp(x)+1)/(np.exp(x)-1))-4
	temperature_tSZ = T_CMB*y*f_x
	
	return temperature_tSZ
	
def planck_beams(nu):
	'''Returns the FWHM of the Planck LFI and HFI form the
	2015 PlancK explanatory supplement. 
	Parameters
	----------
	nu: int
		Planck central band frequency in GHz
	Returns
	-------
	uc: float
		Planck beam FWHM at given frequency
	'''

	beams = {30: 32.29, 44: 27.00, 70: 13.21, 
			 100: 9.68, 143: 7.30, 217: 5.02, 
			 353: 4.94, 545: 4.83, 857: 4.64}
	fwhm = beams[nu]
	
	return(fwhm)

def planck_uc(nu):
	'''Returns the K_CMB --> MJy unit conversion factors form the
	2015 PlancK explanatory supplement. 
	Parameters
	----------
	nu: int
		Planck central band frequency in GHz
	Returns
	-------
	uc: float
		Unit conversion factor at given frequency
	'''

	conversion = {30: 23.5099, 44: 55.7349, 70: 129.1869, 
				100: 244.0960, 143: 371.7327, 217: 483.6874, 
				353: 287.4517, 545: 58.0356, 857: 2.2681}
	uc = conversion[nu]
	return(uc)
