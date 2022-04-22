#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sncosmo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
np.set_printoptions(precision=2)



class Hsiao:
    
    """
    
    Data generation (flux, total flux, noise, total flux + noise) 
    about supernova and lensed supernova thanks to Hsiao model. 
    The purpose is to use these data to train a neural network 
    to recognize/detect strong lensed supernova.
        

    Parameters:
    --------
    nb_images: 'int'
        Number of lensed supernova images 
            
    redshift: 'float'
        Lensed supernova images redshift
        Parameter of the Hsiao model
        
    amplitude: 'float'
        Parameter of the Hsiao model
        (Test 1e-4)
        
    datatype: 'string'
        Define the data on the graph returned by the Graph function.
        ('Flux', 'Total_Flux_Without_Noise', 'Noise', 'Total_Flux_With_Noise')
    
    mu: 'array'  (size = #images)
        Array composed of the magnification value for each supernova image.
    
    dt: 'array'  (size = #images)
        Array composed of time delays.
        
    t0: 'float'
        Time origin. 
    
    noiselevel: 'float' (between 0 and 1)
        Noise level defined as a percentage of the flux.
        
    nobs: 'array' (size = 3 (for the number of bands))
        Number of observations per bands.
    
    dnobs: 'array' (size = 3)
    ddnobs: 'array' (size = 3)
        Array used to define time steps between two differents observations.
    
    """ 
    
#################################################
 
    def __init__(self, nb_images, redshift, amplitude, datatype, mu, dt, t0, noiselevel,  
                 nobs = np.array([91, 90, 30]), dnobs = np.array([1, 1, 4]), ddnobs = np.array([0.8,0.8,1.2]) ):
        
        self.redshift = redshift
        self.nb_images = nb_images
        self.amplitude = amplitude
        self.datatype = datatype
        self.mu = mu
        self.dt = dt
        self.t0 = t0
        self.bands = ('ztfg', 'ztfr', 'ztfi')
        self.pers = noiselevel
        self.nobs = nobs
        self.dnobs = dnobs
        self.ddnobs = ddnobs
        self.model = sncosmo.Model(source = 'hsiao') 
    
#################################################

    def generated_time(self):
        
        """
        Return: 
        -------
        
        ts_image: 'array'  shape (#images, 91, 3) 
            Time (usefull for time delays calculations). 
            Used to generate a time grid used to calculate lightcurves. 
        """

        # time steps between observations
        steps = np.random.poisson(self.dnobs, size = [np.max(self.nobs), len(self.dnobs)]) + self.ddnobs[None]
        t_sample = np.sum(np.tri(np.max(self.nobs))[:, :, None]*steps[None, :, :], axis = 1) + 54965
        # list of indices used to remove non-observed days from band
        indices = np.arange(np.max(self.nobs))[:,None]*np.ones(len(self.nobs))[None,:] 
        t_sample[indices>=self.nobs[None,:]*np.ones(np.max(self.nobs))[:,None]] = None
        # Time samples per band per image with time delays
        ts_image = t_sample[None,:,:] - self.dt[:,None,None] 
        
        return ts_image
     
#################################################
             
    def flux(self):
    
        """
            
        Return:
        --------
        imfluxes: 'array'  shape (#images, 3, 91)
            Flux per image and per band. 
            
        """

        # np.random.seed(202)
        # seedsn = np.random.randint(0,100000) 
        # np.random.seed(seedsn)
        
        self.model.set(z = self.redshift, t0 = self.t0, amplitude = self.amplitude)

        
        imfluxes = np.zeros((self.nb_images, len(self.bands), max(self.nobs)))

        for i in range(self.nb_images):
            for j in range(len(self.bands)):
                imfluxes[i, j, :self.nobs[j]] = self.mu[i] * self.model.bandflux(self.bands[j], 
                                                                                 self.generated_time()[i, :self.nobs[j], j], zp=None)
        
        imfluxes[imfluxes == 0.] = None
             
        return imfluxes
    
    
#################################################   

    
    def total_flux_without_noise(self):
        
        """
        
        Return:
        -------
        TFlux: 'array'   shape (3, 91)
            Total flux per band.
            
        """
        
        TFlux = np.sum(self.flux() , axis=0)
        
        return TFlux
    
#################################################
    
    def generated_noise(self):
        
        """
        Return:
        -------
        Noise: 'array'  shape(3, 91)
            Generated noise proportionnal to 'noise_level' percentage of the maximum flux
        """
        noises = np.full((len(self.bands), max(self.nobs)), self.pers*np.nanmax(self.total_flux_without_noise()))
        Noise = np.random.normal(0, noises)
    
        return Noise
    
#################################################
    
    def total_flux_with_noise(self):
        
        """
        Return:
        -------
        TFluxN: 'array'  shape(3, 91)
            Total flux with noise per band. 
    
        """

        TFluxN = self.generated_noise() + self.total_flux_without_noise()
        
        return TFluxN

#################################################

    def graph(self):
        
        """
        Return:
        -------
        Graph of the chosen data type between flux, total flux (w or w/o noise), noise.
        
        """
        
        fig, ax = plt.subplots()
        for i in range(len(self.bands)):
            
            if (self.datatype == 'Flux'):
                for j in range(self.nb_images):
                    ax.plot( self.generated_time()[j, :, i], self.flux()[j][i][:], label = f'{self.bands[i]} : image {j}')
                    ax.set_ylabel('Flux (photon / s / cm2)')
                    
            elif (self.datatype == 'Total_Flux_Without_Noise'):
                ax.plot(self.generated_time()[2, :, i], self.total_flux_without_noise()[i], label = self.bands[i])
                ax.set_ylabel('Total flux (photon / s / cm2)')
                
            elif (self.datatype == 'Noise'):
                ax.plot(self.generated_time()[2, :, i], self.generated_noise()[i], label = self.bands[i])
                ax.set_ylabel('Noise')
                
            else:
                ax.plot(self.generated_time()[2, :, i], self.total_flux_with_noise()[i], label = self.bands[i])
                ax.set_ylabel('Total flux with noise (photon / s / cm2)')
                
        ax.set_xlabel('Days')
        ax.set_title(f'{self.nb_images} images')
        ax.legend(loc='best')
                
        return ax 


#################################################

    def time_delays(self):
        
        """
        
        Return:
        -------
        df: 'Dataframe'
            Time delays between an image and the next one for each band. 
            They are calculated according to the maximum value reached by the flux. 
            
            Perhaps the calculation mode should be changed, 
            or perhaps time delays should be calculated by interpolating the flux curves
        
        """
        
        TD = np.zeros((self.nb_images -1, len(self.bands)))
        TDBis = np.zeros((self.images -1, len(self.bands)))
        t = self.generated_time()
        f = self.flux()
        
        for i in range(self.nb_images - 1):
            for j in range(len(self.bands)):
                
                index1 = np.argmax(f[i, j, :])
                index2 = np.argmax(f[i+1, j, :])
                TD[i, j] = abs(t[i, j, index1] - t[i+1, j, index2])
                TDBis[i, j] = abs(index1 - index2)
        
        df = pd.DataFrame(TD, columns = ['Band g', 'Band r', 'Band i'])
        
        return df
    
#################################################

    def dataframe(self):
        
        """
        
        Return:
        -------
        
        df_truth: 'dataframe'   shape(1, 8)
            Composed of data used to generate time samples and flux
            
        df_data: 'dataframe'  shape(number of observations, 7)
            Composed  of generated data such as time samples for each bands and total flux with noise
        
        """
        df_truth = pd.DataFrame(
                ["-".join([str(self.nb_images), str(self.amplitude), str(self.redshift), str(self.pers)]),
                self.nb_images, 
                self.t0, 
                self.amplitude,
                str(self.dt), 
                str(self.mu), 
                self.redshift, 
                self.pers], 
                index = [ "ID", "images", "time origin", "amplitude", "time delays", "magnifications", "redshift", "noise level" ]
            )
        
        df_data = pd.DataFrame(data=
            {   "ID" : "-".join([str(self.nb_images),str(self.amplitude), str(self.redshift), str(self.pers)]),
                "time sample band g": self.generated_time()[-1, :, 0],
                "total flux + noise band g": self.total_flux_with_noise()[0], 
                "time sample band r": self.generated_time()[-1, :, 1],
                "total flux + noise band r": self.total_flux_with_noise()[1],
                "time sample band i": self.generated_time()[-1, :, 2],
                "total flux + noise band i": self.total_flux_with_noise()[2],
                }
            )
        return df_truth.T, df_data
        
#################################################

    def dataframe2(self):
        
        """
        
        Return:
        -------
        
        df_truth: 'dataframe'   shape(1, 7)
            Composed of data used to generate time samples and flux
            
        df_data: 'dataframe'  shape(number of observations, 7)
            Composed  of generated data such as time samples for each bands and total flux with noise
        
        """
        df_truth = pd.DataFrame(
                [self.nb_images * self.amplitude * self.redshift* self.pers,
                self.nb_images, 
                self.t0, 
                self.amplitude,
                str(self.dt), 
                str(self.mu), 
                self.redshift, 
                self.pers], 
                index = [ "ID", "images", "time origin", "amplitude", "time delays", "magnifications", "redshift", "noise level" ]
            )
        
        df_data = pd.DataFrame(data=
            {   "ID": self.nb_images * self.amplitude* self.redshift * self.pers,
                "images" : self.nb_images,
                "time sample band g": self.generated_time()[-1, :, 0],
                "total flux + noise band g": self.total_flux_with_noise()[0], 
                "time sample band r": self.generated_time()[-1, :, 1],
                "total flux + noise band r": self.total_flux_with_noise()[1],
                "time sample band i": self.generated_time()[-1, :, 2],
                "total flux + noise band i": self.total_flux_with_noise()[2],
                }
            )
        return df_truth.T, df_data
    
    
    
H=Hsiao(3, 0.4, 1e-4, "Flux", np.array([1.2, 1.42, 1.52]), np.array([0, 10.34, 24.32]), 55000., 0.05)

