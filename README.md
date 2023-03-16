# SubsurfaceSinks
Subsurface Sinks model of contaminant transport in a vegetated treatment system. 

## Installing Required Packages
The following Conda environment contains all required Python packages for running the model. This code has been tested on Windows 10 and 11, some packages may need to be changed for installation on Mac/Unix computers (you can manually install most packages). Installation is typically fairly rapid on a normal desktop computer.  If this doesn't work, see "Method 2" below - this may have package incompatability issues but might work if the conda env fails
 <br>
 ### Method 1
Step 0 - Install Conda. An easy way to do this is by installing [Anaconda](https://www.anaconda.com/), which also contains many Python packages<br>
Step 1 - Clone my github repository from https://github.com/tfmrodge/SubsurfaceSinks<br>
Step 2 - Open a terminal (I use Anaconda Prompt), then navigate to the github repository and run <pre><code> conda env create -f bcenv.yml</code></pre><br>
Step 3 - Run the command <pre><code> conda activate bcenv</code></pre><br>
Step 4 - Run the command <pre><code> pip install hydroeval openpyxl</code></pre><br>
Step 5 - Run the command <pre><code> jupyter notebook</code></pre> to open the list of notebooks, and open the "Bioretention Blues Model Tutorial". You may need to change the kernel (under Kernel>Change Kernel>FugModelEnv) in the Jupyter toolbar, or if you are in the correct environment in your terminal it will bring open with that kernel<br>
### Method 2
Step 0 - Install Conda. An easy way to do this is by installing [Anaconda](https://www.anaconda.com/), which also contains many Python packages<br>
Step 1 - Clone my github repository from https://github.com/tfmrodge/SubsurfaceSinks<br>
Step 2 - Open a terminal with Python (I use Anaconda Prompt), then run <pre><code> conda create -n bcenv python=3.10 pandas joblib seaborn scipy </code></pre><br>
Step 3 - Run the command <pre><code> conda activate bcenv</code></pre><br>
Step 4 - Run the command <pre><code> pip install hydroeval openpyxl</code></pre><br>
Step 5 - Run the command <pre><code> jupyter notebook</code></pre> to open the list of notebooks, and open the "Bioretention Blues Model Tutorial". You may need to change the kernel (under Kernel>Change Kernel>FugModelEnv) in the Jupyter toolbar, or if you are in the correct environment in your terminal it will bring open with that kernel<br>
Step 6 - That's it! Check out the tutorial and enjoy. Please let me know if you have problems

The models are built as class objects. To get information on the class methods, read the code. If you have questions, or notice a bug, or want to say hello, please contact me (Tim Rodgers). My [Google Scholar Page](https://scholar.google.com/citations?user=npsj5x4AAAAJ&hl=en&oi=ao) should have my current email.

## BioretentionBlues
Currently implemented with the BioretentionBlues subclass, which is parameterized for a bioretention cell as per Rodgers et al (2022). Read the paper here https://doi.org/10.1021/acs.est.1c07555

You can also run the BioretentionBlues subclass as parameterized for the Pine and 8th st system in Vancouver, Canada as presented in Rodgers et al. (in prep).
For this work, we introduced a few new capabilities, including the ability to run the model across intensity-duration-frequency curves. These are currently defined for the City of Vancouver, running them efficiently uses the joblib parallelization package which can be found here: https://joblib.readthedocs.io/en/latest/

## WastewaterWetland
Implementation for a subsurface flow wastewater treatment wetland. Currently paramterized for the Oro Loma Horizontal Levee system in San Leandro, California.
