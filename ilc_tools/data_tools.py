import numpy as np
from astropy.io import fits
from astropy.io import ascii

def writefits(file_name, data, header=None):
    '''Writes a 2D array (usually an image) to a fits file.
    
    Parameters
    ----------
        file_name: string
            Name of the fits file
        data: float
            2D array to be written
        header: fits header
            Optional fits header. A minimal header is created
            automatically if none is provided. Default: None    

    Returns
    -------
    None
    '''
    
    hdu = fits.PrimaryHDU()
    hdu.data = np.array(data, dtype=np.float32)

    if header is not None:    
        hdu.header = header
    hdu.writeto(file_name, overwrite=True)    
    
    return(None)

    
def readfits(file_name):
    '''Reads fits files and returns data array and header.   
 
    Parameters
    ----------
        file_name: string
            Name of the fits file    

    Returns
    -------
        data: float array
            Data array extracted from the fits file
        header: fits header
            Header of the fits file
    ''' 
    
    hdulist=fits.open(file_name)
    data = hdulist[0].data
    header = hdulist[0].header
    hdulist.close()    
    
    return(data, header) 

def write_file(file_name, data, names = None, overwrite=True):
    '''Writes a multi-dimensional array or table object to an ascii file
    with fixed-width columns and no delimiter.
       
    Parameters
    ----------
        file_name: string
            Name of the ascii file
        data: float array or table object
            Data array or table
        names: string array, optional
            Optional column names. Default: None
        overwrite: bool, optional
            If set to True, files with identical names will be overwritten.
            Default: True

    Returns
    -------
    None
    '''

    ascii.write(data, file_name, format='fixed_width',  delimiter_pad=' ', 
    delimiter=None, fast_writer=False, overwrite=overwrite, names=names)
    return(None)

def read_file(file_name):
    '''Reads ascii files and returns data as table object.
     
    Parameters
    ----------
        file_name: string
            Name of the ascii file
      
    Returns
    -------
        data: table object
            Table object containing the read data
    '''

    data = ascii.read(file_name)
    return(data)

