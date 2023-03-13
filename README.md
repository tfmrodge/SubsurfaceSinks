# SubsurfaceSinks
Subsurface Sinks model of contaminant transport in a vegetated treatment system. 

## Installing Required Packages
The following Conda environment contains all required Python packages for running the model. This code has been tested on Windows 10 and 11, some packages may need to be changed for installation on Mac/Unix computers (you can manually install most packages). Installation is typically fairly rapid on a normal desktop computer.
 <br>
Step 0 - Install Conda. An easy way to do this is by installing [Anaconda](https://www.anaconda.com/), which also contains many Python packages<br>
Step 1 - Clone my github repository from https://github.com/tfmrodge/SubsurfaceSinks<br>
Step 2 - Open a terminal (I use Anaconda Prompt), then navigate to the github repository and run <pre><code> conda env create -f bcenv.yml</code></pre><br>
Step 3 (optional) - Run the command <pre><code> conda activate bcenv</code></pre><br>
Step 4 - Run the command <pre><code> jupyter notebook</code></pre> to open the list of notebooks, and open the "Bioretention Blues Model Tutorial". You may need to change the kernel (under Kernel>Change Kernel>FugModelEnv) in the Jupyter toolbar, or if you are in the correct environment in your terminal it will bring open with that kernel<br>
Step 5 - That's it! If you want to use the "run_IDFs.py" script, you will need to install the "JobLib" package as well (or run non-parallel). To do so, refer to the documentation for JobLib (this is a more advanced feature, contact me if you need help)

The models are built as class objects. To get information on the class methods, read the code. If you have questions, or notice a bug, or want to say hello, please contact me (Tim Rodgers). My [Google Scholar Page](https://scholar.google.com/citations?user=npsj5x4AAAAJ&hl=en&oi=ao) should have my current email.

## BioretentionBlues
Currently implemented with the BioretentionBlues subclass, which is parameterized for a bioretention cell as per Rodgers et al (2022). Read the paper here https://doi.org/10.1021/acs.est.1c07555

You can also run the BioretentionBlues subclass as parameterized for the Pine and 8th st system in Vancouver, Canada as presented in Rodgers et al. (in prep).
For this work, we introduced a few new capabilities, including the ability to run the model across intensity-duration-frequency curves. These are currently defined for the City of Vancouver, running them efficiently uses the joblib parallelization package which can be found here: https://joblib.readthedocs.io/en/latest/

## WastewaterWetland
Implementation for a subsurface flow wastewater treatment wetland. Currently paramterized for the Oro Loma Horizontal Levee system in San Leandro, California.
