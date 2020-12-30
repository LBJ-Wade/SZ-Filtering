import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import ascii 


cosmo = FlatLambdaCDM(H0= 70, Om0 = 0.3)
H0 = cosmo.H0.value

def radial_profile(data, pxsize, return_k = False):
    '''Computes the azimuthally-averaged radial profile of a given map.
    Parameters
    ----------
    pxsize: float
      Pixel size in arcmin.
    data: 2D float or complex array
      Input image.
    return_k: bool, optional
      If set to True, the provided image is assumed to be a power spectrum
      and the x-coordinate of the returned data is converted to the 
      two-dimensional spatial frequency k. Default: None    
    Returns
    -------
    kradialprofile: float array
      x-coordinate of the radial profile. Either the radial separation 
      from the image center or spatial frequency k, depending on the 
      value of the variable return_k.
    radialprofile: float or complex array
      azimuthally averaged radial profile.
    ''' 
    
    npix = data.shape[1]
    center = (npix/2,npix/2)
    y, x = np.indices((data.shape)) # first determine radii of all pixels
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2) #compute distance
    
    if return_k is True:
        k = 360/(pxsize/60.)*np.sqrt(((x - center[0])/npix)**2 + ((y - center[1])/npix)**2) # section 2.3 POKER paper
    else:
        k = np.copy(r)
        
    r = r.astype(np.int) #bincount takes only integer type
    
    tbin_r = np.bincount(r.ravel(), np.real(data.ravel()))
    tbin_i = np.bincount(r.ravel(), np.imag(data.ravel())) # sum for image values in radius bins
    
    nr = np.bincount(r.ravel()) # number of pixels in radius bin
    
    kradialprofile = np.bincount(r.ravel(), k.ravel()) / nr
    radialprofile_r = tbin_r / nr  # compute average for each bin
    radialprofile_i = tbin_i / nr 
    radialprofile = radialprofile_r + radialprofile_i*1j
    
    return (kradialprofile, radialprofile) 
    
def power_spec(image, pxsize, return_k = False):
    '''Computes the azimuthally-averaged power spectrum of a given map.
	Parameters
	----------
	image: 2D float array
		Input image
	pixel_size_arcmin: float
		Pixel size in arcmin.
	return_k: bool, optional
		If set to True, the provided image is assumed to be a power spectrum
		and the x-coordinate of the returned data is converted to the 
		two-dimensional spatial frequency k. Default: None	
	Returns
	-------
	k: float array
		x-coordinate of the radial profile. Either the radial separation 
		from the image center or spatial frequency k, depending on the 
		value of the variable return_k.
	Pk: float or complex array
		azimuthally-averaged power spectrum
    ps: float or complex 2D array
        2D power spectrum of the image
	''' 
    nrpix = image.shape[0] * image.shape[1]
    
    F_k = np.fft.fft2(image, norm = None)
    Fk = np.fft.fftshift(F_k) / nrpix # divide by npix to ensure the fft is always identical and  Now shift the quadrants around so that low spatial frequencies are in the center of the 2D fourier transformed image.
    
    ps=(np.absolute((Fk))**2) # Calculate 2D power spectrum
    
    k, Pk = radial_profile(ps, pxsize, return_k=return_k)  # Calculate the azimuthally averaged 1D power spectrum. k starts from 0 so beware. radial profile computed from radius of circle and not map boundary
    
    return(k, Pk, ps)

def fit_data(array, npix, bins, fit = False, plot = False, xlabel = "x"):
    '''Creates a histogram with evenly spaced bins and fits a Guassian.
    
    Parameters:
    -----------
        estimated_map: float array
            2D data array used to create a histogram
        npix: int
            number of pixels
        bins: int
            number of bins to be used
        fit: bool; optional
            If True, a guassian is fit to the data
        plot: bool, optional
            If True, the histogram is plot
        xlabel: string, optional
            X-label to be used when creating a plot. Default: "x"
            
    Returns:
    --------
        popt: float array
            Default value is a flaot array with 2 columns, one for hostpgram values and the other for bin centres. If fit is set to True, the function will instead return the best-fit parameters for a Gaussian fitted to the histogram.
    '''
    
    hist, bin_edges = np.histogram(array, bins=bins)
    bin_centres = (bin_edges[0:-1] + bin_edges[1:])/2 #subtract by median to correct for offset

    #Guassian function to be fit to the data
    def gaussian(x, *p):
        a, mu, sigma = p #a = 1/(sigma*np.sqrt(2*np.pi))
        gauss = a*np.exp((-1/2)*((x- mu)/sigma)**2) 
        return gauss

    
    if fit is True:
        popt, pcov = curve_fit(gaussian, bin_centres, hist, p0=(np.max(hist), np.mean(array), np.std(array)), sigma = np.sqrt(hist+1)) # p0 is the initial guess for the fitting coefficients

        hist_fit = gaussian(bin_centres, *popt)

    if plot is True:
        axes = plt.gca()
        axes.set_ylim(1) 
        axes.set_yscale('log')
        plt.setp(axes.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.hist(array.reshape(npix**2), bins, color = '0.75')
        
        if fit is True:
            plt.plot(bin_centres, hist_fit, 'k')
            print('Mean fit = ', popt[1])
            print('Standard Deviation fit = ', popt[2])

        plt.xlabel(xlabel)
        plt.ylabel("No. of pixels")
        plt.show()
        
    if fit is True:
        return(popt)
    
    else:
        return(hist, bin_centres)

def pxsize_compute(data_array, n):
    '''Computes the pixel size for each cluster given the physical size and redshift
    Parameters:
    ----------
        data_array: float array
            multi-dimensional data array consititing of rows of clusters
            and columns with redshift and R_500
        n: float point
            arbitary number chosen to be same for all clusters.
            
    Returns:
    --------
        pxsize_array: float array
            pixel size in arc minutes for given number of clusters 
    '''
    
    pxsize_array = [] #in arcmin
    theta_array = [] #in arcmin
    for i in np.arange(len(data_array)):
        z = data_array[i,2]
        R_500 = data_array[i,3]
        theta = (R_500)/ (cosmo.angular_diameter_distance(z).si.value) * (180 / np.pi) *60 #convert from radians to arcmin R_500 in Mpc and angular diameter distance also in Mpc
        theta_array.append(theta)
        #print(theta_array)
        pxsize = theta*n #recommended n is 0.05
        pxsize_array.append(pxsize)
    return np.array(pxsize_array)
