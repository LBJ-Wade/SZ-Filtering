import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import healpy as hp
from ilc_tools import data_tools as dt
from ilc_tools import misc, sz


def project_maps(npix, input_file = None, RA = None, DEC = None, pxsize = None, same_units = False, same_res = False):
    '''Project HEALPIX maps using gnomic view given coordinates.
    
    Parameters:
    -----------
        npix: int
            number of pixels
        input_file: str, optional
            Name of the data file to be used containing RA,DEC and pixel size.
        RA: float array, optional
            Right acention of objects, ICRS coordinates are required. Default:None
        DEC: float array, optional
            Declination of objects, ICRS coordinates are required. Default:None
        pxsize: float, optional
            pixel size in arcminutes. REcommended: 1.5
        same_units: bool, optional
            if changed to True all Planck maps will be provided in units of K_CMB.
            Default: False
        same_res: bool, optional
            if changed to True all Planck maps will be provided with the resolution 
            of the lowest-frequency channel. Default: False
        
    Returns:
    --------
        output: array
            single image or data cube containing the projected maps. 
            If out_path is set, one or several files will be written
    '''

    
    if input_file is not None:
        data_new = dt.ascii.read(input_file)
        ra = np.array(data_new[:]['RA'])
        dec = np.array(data_new[:]['DEC'])
        pixsize = np.array(data_new[:]['pixel_size'])
        nclusters = len(ra)
    else:
        ra = RA
        dec = DEC
        pixsize = pxsize
        nclusters = 1
    
    freq = [143, 217, 353, 545, 857] #0-353GHz are in K_cmb while 545 and 857GHz are in MJy/sr 
    nf = len(freq)
    
    A = (2*np.sqrt(2*np.log(2)))
    
    output = np.zeros((nclusters, nf, npix, npix))
    
    for i in np.arange(nclusters):
        
        for f in np.arange(nf):
            all_sky = hp.read_map('HFI_{0}_layer.fits'.format(f)) 
            projected_map = hp.gnomview(all_sky, coord=('G','C'), rot=(ra[i],dec[i]), return_projected_map=True, xsize=npix, reso=pixsize[i], no_plot = True)
            
            if same_units is True: #from https://wiki.cosmos.esa.int/planckpla2015/index.php/UC_CC_Tables
                if f == 0:
                    projected_map *= 371.7327
                if f == 1:
                    projected_map *= 483.6874
                if f == 2:
                    projected_map *= 287.4517
                    
       
            if same_res is True and f != 0:
                kernel = np.sqrt(sz.planck_beams(freq[0])**2 - sz.planck_beams(freq[f])**2)
                print(sz.planck_beams(freq[0]), sz.planck_beams(freq[f]), kernel/A/pixsize[i])
                projected_map = ndimage.gaussian_filter(projected_map, sigma= kernel/A/pixsize[i], order=0, mode = "reflect", truncate = 10)
                
            output[i,f,:,:] = projected_map
            
    print(output.shape)
    return(output)      

def offset_correction(estimated_map, npix = 400, bins = 300,  median = False, gaussian = False, fit = False, plot = False):
    '''Corrects the offset present in the ILC maps.
    Parameters:
    -----------
        estimated_map: float array
            ILC map obtained after computing weights
        npix: int
            number of pixels. Default is 400
        bins: int
            number of bins
        median: bool, optional
            Subtracts the median from the estiamted map if True.
        gaussian: bool, optional
            If True, it fits a gaussian to the histogram and 
            subtracts the best fit centre.
    
    Returns:
    --------
        estimated_map: float array
            offset corrected ILC map
    '''
    if median is True:
        estimated_map = estimated_map - np.median(estimated_map)
        
    if gaussian is True:
        popt =  misc.fit_data(estimated_map, npix, bins, fit = True, plot = False)
        estimated_map = estimated_map - popt[0]
        print('Best mean- fit', p_opt[0])

    return(estimated_map)
    
    
def ilc_run(data,F = None, e = None, constrain = None, offset = False, bins = None ):
    '''Runs an internal linear combination on a given set of multi-frequency maps
       
    Parameters
    ----------
        data: float array
            a 3D array of dimensions n_freq x npix x npix. 
        F: array
            spectral information of the desired map, either CMB or tSZ. The 
            dimensions have to be n_components x n_freq
        e: array, optional
            If multible spectral components are constrained, e gives the
            response of the ilc weights to the individual spectra
        constrain: array
            Spectral constraints for the ilc algorithm. If contaminants
            are constrained as well, the dimensions have to be 
            n_components x n_freq
        offset: bool, optional
            Default: True. Subtracts the median of the data from the estimated_map.
            Gaussian can be fit and best mean fit subtracted if 'median' is replaced by 'gaussian'. 
        bins: int
            number of bins not more than the number of pixels. 
        
    Returns
    --------
        reconstructed_map: array
            Compton y map or a CMB map that has dimensions npix x npix
    '''

    nf = data.shape[0]
    npix = data.shape[1]

    matrix = np.cov(data.reshape(nf,npix*npix))
    cov_inverse = np.linalg.inv(matrix) 

    if e is None: #standard ilc
        weights = (cov_inverse@F)/(np.transpose(F)@cov_inverse@F)
        print(weights)

    else: #constrained ilc
        X = np.array([F,constrain]) 
        weights = np.transpose(e)@(np.linalg.inv(X@cov_inverse@np.transpose(X)))@X@cov_inverse
        print(weights)

    reconstructed_map = ((weights[0]*data[0]) + (weights[1]*data[1]) + (weights[2]*data[2]) + (weights[3]*data[3]) + (weights[4]*data[4]))

    if offset is True: 
        reconstructed_map = offset_correction(reconstructed_map, npix, bins, gaussian = True)

    return(reconstructed_map)
