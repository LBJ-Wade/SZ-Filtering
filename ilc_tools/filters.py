import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from ilc_tools import misc
from scipy import interpolate
from scipy import ndimage
from scipy import signal
from pymf import make_filter_map 
from ilc_tools import data_tools
import nifty5 as ift 

def radialprofile_cmb(data, rmax=100, nbins=20):
    
    npix = data.shape[1]
    center = (npix/2,npix/2)
    y, x = np.indices((data.shape[1], data.shape[1])) # first determine radii of all pixels
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2) #compute distance
    
    bins = np.linspace(0, rmax, nbins)
    
    nc = data.shape[0]
    rad_profile = np.zeros((nc,2, nbins-1))
    error = np.zeros((nbins-1))
    
    for f in np.arange(nc):

        for i in np.arange(len(bins)-1):
            index = (r>bins[i]) & (r <bins[i+1])
            rad_profile[f,0,i] = (bins[i] + bins[i+1])/2
            rad_profile[f,1,i] = np.mean(data[f,index])
        
    for i in np.arange(len(bins)-1):
        error[i] = np.std(rad_profile[:,1,i])
              
    return rad_profile, error
    
def radialprofile_ksz(data, rmax=100, nbins=20):
    
    npix = data.shape[1]
    center = (npix/2,npix/2)
    y, x = np.indices((data.shape[1], data.shape[1])) 
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2) 
    
    bins = np.linspace(0, rmax, nbins)

    rad_profile = np.zeros((2, nbins-1))
    error = np.zeros((nbins-1))
    

    for i in np.arange(len(bins)-1):
        index = (r>bins[i]) & (r <bins[i+1])
        rad_profile[0,i] = (bins[i] + bins[i+1])/2
        rad_profile[1,i] = np.mean(data[index])
        
    return rad_profile


def WienerFilter(data, signal_template, pixel_size, smooth = True, FWHM = 9.68, pxsize = 0.2, npix = 400):
    
    hdulist = fits.open(signal_template)
    ksz = hdulist[0].data
    header = hdulist[0].header
    hdulist.close()
    
    if smooth is True:
        sigma_beam = FWHM / (2*np.sqrt(2*np.log(2))) / pxsize
        ksz = ndimage.gaussian_filter(ksz, sigma=sigma_beam, order=0, mode='wrap', truncate=20.0)
        
    signal_fft = misc.power_spec(ksz, pxsize, True)
    signal_fft_nok = misc.power_spec(ksz, pxsize, False)
    
    hdulist = fits.open(data)
    data_map = hdulist[0].data
    header = hdulist[0].header
    hdulist.close()
    
    pixisize = data_tools.ascii.read(pixel_size)
    pixsize = np.array(pixisize[:]['pixel_size']).tolist()
    
    nclusters = data_map.shape[0]
    
    
    filtered_maps = []
    filters = []
    
    for i in np.arange(nclusters):
    
        noise_fft = misc.power_spec(data_map[i], pxsize = pixsize[i], return_k = True) #create power spec
        noise_fft_nok = misc.power_spec(data_map[i], pxsize = pixsize[i], return_k = False)

        signal_ps = np.interp(noise_fft[0], signal_fft[0], signal_fft[1]) #interpolate

        noise_ps_map = make_filter_map(data_map[i], noise_fft_nok[0], noise_fft_nok[1]) #make filter maps
        signal_ps_map = make_filter_map(data_map[i], noise_fft_nok[0], signal_ps)

        filter_window = signal_ps_map / (signal_ps_map + noise_ps_map)
        #plt.semilogy(np.real(filter_window[200,200:]))

        data_fft = np.fft.fftshift(np.fft.fft2(data_map[i], norm = None)/ npix**2)
        filtered_image = np.real(np.fft.ifft2(np.fft.ifftshift(filter_window*data_fft))) * npix**2

        filtered_maps.append(filtered_image)
        filters.append(filter_window)

    return (filtered_maps, filters)

def nifty_wf(signal, noise, y_map, npix = 400, pxsize = 1.5, kernel = 9.68, n = 10, smooth = False):
    
    cmb_mocks = noise.shape[0]
    
    A = (2*np.sqrt(2*np.log(2)))
    
    if smooth is True:
        signal_smooth = np.zeros((cmb_mocks, npix, npix))
        noise_smooth = np.zeros((cmb_mocks, npix, npix))
        
        for i in np.arange(cmb_mocks):
            noise_data = ndimage.gaussian_filter(noise[i], sigma= kernel/A/pxsize, order=0, mode = "reflect", truncate = 10)
            #signal_data = ndimage.gaussian_filter(signal[i], sigma= kernel/A/pxsize, order=0, mode = "reflect", truncate = 10)
            signal_data = signal[i] #uncomment here if smoothing signal and noise
            noise_smooth[i,:,:] = noise_data
            signal_smooth[i,:,:] = signal_data
    else:
        noise_smooth = noise
        signal_smooth = signal
                
    pixel_space = ift.RGSpace([npix, npix]) 
    fourier_space = pixel_space.get_default_codomain()

    s_data = np.zeros((cmb_mocks, npix, npix))
    m_data = np.zeros((cmb_mocks, npix, npix))
    d_data = np.zeros((cmb_mocks, npix, npix))


    for i in np.arange(cmb_mocks):
        
        signal_field = ift.Field.from_global_data(pixel_space, signal_smooth.astype(float)) #[i] for mock_data
        HT = ift.HartleyOperator(fourier_space, target=pixel_space) 
        power_field = ift.power_analyze(HT.inverse(signal_field), binbounds=ift.PowerSpace.useful_binbounds(fourier_space, True))
        Sh = ift.create_power_operator(fourier_space, power_spectrum=power_field) 
        R = HT
           
        noise_field = ift.Field.from_global_data(pixel_space, noise_smooth[i].astype(float))
        noise_power_field = ift.power_analyze(HT.inverse(noise_field), binbounds=ift.PowerSpace.useful_binbounds(fourier_space, True))

        N = ift.create_power_operator(HT.domain, noise_power_field)
        N_inverse = HT@N@HT.inverse

        data = signal_field + noise_field # --->when using mock_data

        # Wiener filtering the data

        j = (R.adjoint @N_inverse.inverse)(data)
        D_inv = R.adjoint @ N_inverse.inverse @ R + Sh.inverse

        IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
        D = ift.InversionEnabler(D_inv, IC, approximation=Sh.inverse).inverse
        m = D(j)

        #s_data[i,:,:] = (signal_field).to_global_data()
        m_data[i,:,:] = HT(m).to_global_data()
        #d_data[i,:,:] = data.to_global_data()    
    
    #Squaring the filtered map and also taking the absoute val of filtered map
       
    
    # uncomment here for no cross correlation 
    squared_m_data = np.zeros((cmb_mocks, npix, npix))
    abs_m_data = np.zeros((cmb_mocks, npix, npix))
    
    for i in np.arange(m_data.shape[0]):
        squared_m_data[i,:,:]  = m_data[i,:,:] * m_data[i,:,:]
        abs_m_data[i,:,:] = np.abs(m_data[i,:,:])
    
    #Stacking all filtered maps
    stack1  = np.sum(squared_m_data, axis = 0)/m_data.shape[0]
    stack2  = np.sum(abs_m_data, axis = 0)/m_data.shape[0]
       
    return (m_data, squared_m_data, abs_m_data, stack1, stack2) #change here to return the right values ---->, stack_square, stack_abs

'''

    #Stacking progressively
    stack_maps = np.zeros((npix,npix))
    stack_square = np.zeros((int(m_data.shape[0]/n), npix, npix))
    
    k = 0
    for i in np.arange(m_data.shape[0]):
        stack = stack_maps + squared_m_data[i]
        stack_maps[:,:] = stack
        if np.mod(i,n) == 0:
            stack_square[k,:,:] = stack_maps[:,:]
            k = k+1   
            

    stack_abs_maps = np.zeros((npix,npix))
    stack_abs = np.zeros((int(m_data.shape[0]/n), npix, npix))
    
    l = 0
    for i in np.arange(m_data.shape[0]):
        stack = stack_abs_maps + abs_m_data[i]
        stack_abs_maps[:,:] = stack
        if np.mod(i,n) == 0:
            stack_abs[l,:,:] = stack_abs_maps[:,:]
            l = l+1 
  
# cross correlating filtered_map(m_data) with y_map         
        
    squared_corr_data = np.zeros((cmb_mocks, npix, npix))
    abs_corr_data = np.zeros((cmb_mocks, npix, npix))
    
    for i in np.arange(m_data.shape[0]):
        from scipy import signal
        corr_data = signal.correlate(y_map, m_data[i], mode = 'same', method = 'fft')
        squared_corr_data[i,:,:]  = corr_data * corr_data
        abs_corr_data[i,:,:] = np.abs(corr_data)
        
    stack1_corr  = np.sum(squared_corr_data, axis = 0)/m_data.shape[0]
    stack2_corr  = np.sum(abs_corr_data, axis = 0)/m_data.shape[0] 
    
    stack_maps = np.zeros((npix,npix))
    stack_square_corr = np.zeros((int(m_data.shape[0]/n), npix, npix))
    
    k = 0
    for i in np.arange(m_data.shape[0]):
        stack = stack_maps + squared_corr_data[i]
        stack_maps[:,:] = stack
        if np.mod(i,n) == 0:
            stack_square_corr[k,:,:] = stack_maps[:,:]
            k = k+1   
            

    stack_abs_maps = np.zeros((npix,npix))
    stack_abs_corr = np.zeros((int(m_data.shape[0]/n), npix, npix))
    
    l = 0
    for i in np.arange(m_data.shape[0]):
        stack = stack_abs_maps + abs_corr_data[i]
        stack_abs_maps[:,:] = stack
        if np.mod(i,n) == 0:
            stack_abs_corr[l,:,:] = stack_abs_maps[:,:]
            l = l+1 
'''

#Simple Wiener Filter

def wf(signal, noise, signal_boost, npix = 400):
    
    pixel_space = ift.RGSpace([npix, npix]) 
    fourier_space = pixel_space.get_default_codomain()

    signal_field = ift.Field.from_global_data(pixel_space, signal.astype(float))
    
    HT = ift.HartleyOperator(fourier_space, target=pixel_space) 
    power_field = ift.power_analyze(HT.inverse(signal_field), binbounds=ift.PowerSpace.useful_binbounds(fourier_space, True))

    Sh = ift.create_power_operator(fourier_space, power_spectrum=power_field) 
    R = HT
 
    noise_field = ift.Field.from_global_data(pixel_space, noise.astype(float))
    noise_power_field = ift.power_analyze(HT.inverse(noise_field), binbounds=ift.PowerSpace.useful_binbounds(fourier_space, True))

    N = ift.create_power_operator(HT.domain, noise_power_field)
    N_inverse = HT@N@HT.inverse
    
    amplify = len(signal_boost)
    
    s_data = np.zeros((amplify, npix, npix))
    m_data = np.zeros((amplify, npix, npix))
    d_data = np.zeros((amplify, npix, npix))

    for i in np.arange(amplify):
        
        data = signal_field * signal_boost[i] +  noise_field #

        # Wiener filtering the data

        j = (R.adjoint @N_inverse.inverse)(data)
        D_inv = R.adjoint @ N_inverse.inverse @ R + Sh.inverse

        IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
        D = ift.InversionEnabler(D_inv, IC, approximation=Sh.inverse).inverse
        m = D(j)

        s_data[i,:,:] = (signal_field * signal_boost[i]).to_global_data()
        m_data[i,:,:] = HT(m).to_global_data()
        d_data[i,:,:] = data.to_global_data()

    return (s_data, m_data, d_data)

def wf_test(signal, noise, signal_boost, npix = 400):
    
    pixel_space = ift.RGSpace([npix, npix]) 
    fourier_space = pixel_space.get_default_codomain()

    signal_field = ift.Field.from_global_data(pixel_space, signal.astype(float))
    
    HT = ift.HartleyOperator(fourier_space, target=pixel_space) 
    power_field = ift.power_analyze(HT.inverse(signal_field), binbounds=ift.PowerSpace.useful_binbounds(fourier_space, True))

    Sh = ift.create_power_operator(fourier_space, power_spectrum=power_field) 
    R = HT
 
    noise_field = ift.Field.from_global_data(pixel_space, noise.astype(float))
    noise_power_field = ift.power_analyze(HT.inverse(noise_field), binbounds=ift.PowerSpace.useful_binbounds(fourier_space, True))

    N = ift.create_power_operator(HT.domain, noise_power_field)
    N_inverse = HT@N@HT.inverse
    
    amplify = len(signal_boost)
    
    s_data = np.zeros((amplify, npix, npix))
    m_data = np.zeros((amplify, npix, npix))
    d_data = np.zeros((amplify, npix, npix))

    for i in np.arange(amplify):
        
        data = noise_field 

        # Wiener filtering the data

        j = (R.adjoint @N_inverse.inverse)(data)
        D_inv = R.adjoint @ N_inverse.inverse @ R + Sh.inverse

        IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-3)
        D = ift.InversionEnabler(D_inv, IC, approximation=Sh.inverse).inverse
        m = D(j)

        s_data[i,:,:] = (signal_field * signal_boost[i]).to_global_data()
        m_data[i,:,:] = HT(m).to_global_data()
        d_data[i,:,:] = data.to_global_data()

    return (s_data, m_data, d_data)
