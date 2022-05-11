# Supernova data generetors 


### Supernova data 

Python scripts for the generation of supernova data from existing models. 

Some data represent 'normal' supernova, and others aim to simulate strong lensed supernova images. 
The obtain results depend on the given number of images. As results there are:
- the luminosity fluxes for each images and depend on the Rubin filter band
- the total flux without noise which is the sum of the luminosity fluxes for each band.
- the total flux with noise which correspond to the total flux without noise + a generated noise. 
- plots of the chosen flux in function of time
- the calculated time-delays between an image and the next one (if there is more than 1 image) for each band. 




### Purpose 

Use these data to train a neural network to recognize/detect strong lensed supernova. 




### Setup

You need Python 3.9.
The project depends on:
- the 'sncosmo',
- the 'numpy',
- the 'matplotlib.pyplot',
- the 'random',
- the 'panda' libraries, 
which can be installed with pip. 




### Contact

Email 'lea.peligry@etu.uca.fr' 





### Contributing

You are welcome to contribute to the code via pull requests.















