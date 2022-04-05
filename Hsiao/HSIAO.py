#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sncosmo
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
np.set_printoptions(precision=2)



class Hsiao:
    
    """
    Parameters:
    --------
    images: 'int'
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
        
    Returns: 
    --------
    Data generation (flux, total flux, noise, total flux + noise) 
        about supernova and lensed supernova. 
    
    """ 
    
#################################################
    
    def __init__(self, images, redshift, amplitude, datatype):
        
        self.t0 = 55000.
        self.bands = ('ztfg', 'ztfr', 'ztfi')
        self.pers = 5.0/100
        self.nobs = [91, 91, 91]   #/!\ ds code nb_obs
        self.dnobs = [1, 1, 4]
        self.ddnobs = [0.8,0.8,1.2]
        self.model = sncosmo.Model(source = 'hsiao')
        
        self.redshift = redshift
        self.images = images
        self.amplitude = amplitude
        self.datatype = datatype
        
#################################################        
        
    def generated_dt(self):
        
        """
        
        Return:
        -------
        dt: 'array'  shape(#images)
            Absolute (unobservable) times
            
        """
        
        dt = np.zeros(self.images)
        for i in range(len(dt)-1):
            dt[i+1] = dt[i] + random.SystemRandom().uniform(5, 15)
        
        return dt
    
#################################################
    
    def generated_mu(self):
        
        """
        
        Return:
        -------
        mu: 'array'  shape(#images)
            Magnification for each image
        
        """
        
        mu = np.zeros(self.images)
        for i in range(self.images):
            mu[i] = random.SystemRandom().uniform(1, 1.5)
        
        return mu

#################################################

    def generated_time(self):
        
        """
        Return: 
        -------
        
        time: 'array'  shape (#images, 3, 91)
            Time (usefull for time delays calculations). 
        """
        
        obs = -1*np.ones((len(self.bands), max(self.nobs)))
        time = np.zeros((self.images, len(self.bands), max(self.nobs)))
        
        for i in range(len(self.bands)):
            obsv = np.random.poisson(self.dnobs[i], size = self.nobs[i]) + self.ddnobs[i]
            for j in range(1, self.nobs[i]):
                obsv[j] = obsv[j-1] + obsv[j]
            obs[i, :self.nobs[i]] = obsv + 54965
            
            for k in range(self.images):
                time[k, i, :self.nobs[i]] = obs[i, :self.nobs[i]] - self.generated_dt()[k]
            
        #time = time.remove(0)
            
        return time
    

#################################################
             
    def flux(self):
    
        """
            
        Return:
        --------
        imfluxes: 'array'  shape (#images, 3, 91)
            Flux per image and per band. 
            
        """

        #np.random.seed(202)
        #seedsn = np.random.randint(0,100000)
        #np.random.seed(seedsn)
        
        imfluxes = np.zeros((self.images, len(self.bands), max(self.nobs)))
        self.model.set(z = self.redshift, t0 = self.t0, amplitude = self.amplitude)
        
        for j in range(len(self.bands)):
            
            for i in range(self.images):
                
                imfluxes[i,j,:self.nobs[j]] = self.generated_mu()[i] * self.model.bandflux(self.bands[j], self.generated_time()[i, j, :self.nobs[j]])
            
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
    
    def noise(self):
        
        """
        Return:
        -------
        Noise: 'array'  shape(3, 91)
            Generated noise proportionnal to 5% of the maximum flux
        """
        
        Noises = np.zeros((len(self.bands), max(self.nobs)))
        
        for j in range(len(self.bands)):
            Noises[j,:] = np.full(max(self.nobs), self.pers * np.max(self.total_flux_without_noise()[j][:])) 
            Noise=np.random.normal(0,Noises)
    
        return Noise
    
#################################################
    
    def total_flux_with_noise(self):
        
        """
        Return:
        -------
        TFluxN: 'array'  shape(3, 91)
            Total flux with noise per band. 
    
        """
        
        TFluxN = np.zeros((len(self.bands), max(self.nobs)))
        TFluxN = self.noise() + self.total_flux_without_noise()
        
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
                for j in range(self.images):
                    ax.plot(self.flux()[j][i][:], label = f'{self.bands[i]} : image {j}')
                    ax.set_ylabel('Flux (photon / s / cm2)')
                    
            elif (self.datatype == 'Total_Flux_Without_Noise'):
                ax.plot(self.total_flux_without_noise()[i], label = self.bands[i])
                ax.set_ylabel('Total flux (photon / s / cm2)')
                
            elif (self.datatype == 'Noise'):
                ax.plot(self.noise()[i], label = self.bands[i])
                ax.set_ylabel('Noise')
                
            else:
                ax.plot(self.total_flux_with_noise()[i], label = self.bands[i])
                ax.set_ylabel('Total flux with noise (photon / s / cm2)')
                
        ax.set_xlabel('Days')
        ax.set_title(f'{self.images} images')
        ax.legend()
                
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
        
        TD = np.zeros((self.images -1, len(self.bands)))
        TDBis = np.zeros((self.images -1, len(self.bands)))
        t = self.generated_time()
        f = self.flux()
        
        for i in range(self.images - 1):
            for j in range(len(self.bands)):
                
                index1 = np.argmax(f[i, j, :])
                index2 = np.argmax(f[i+1, j, :])
                TD[i, j] = abs(t[i, j, index1] - t[i+1, j, index2])
                TDBis[i, j] = abs(index1 - index2)
        
        df = pd.DataFrame(TD, columns = ['Band g', 'Band r', 'Band i'])
        
        return df

    
        