## Homology and Geometry of the Plaquette Eden Model


DEVELOPERS: <br />
Anna Krymova (anna.krymova@tum.de) <br />
Erika Roldan (erika.roldan@ma.tum.de) <br />


DATE: MAY 15, 2021

LICENSE: GNU GENERAL PUBLIC LICENSE (see LICENSE)

## Overview 
This package includes software to run simulations of the Plaquette Eden Model, which is a discrete 2-dimensional stochastic cell growth process defined on the 3-dimensional regular cubical tessellation of the Euclidian space.  The sofrware analyzes the topology (Betti numbers and persistent homology) and local geometry of the structure. 
The program also has visualization capabilities and can produce a picture of the Plaquette Eden Model or output a .txt file that can be used with Autodesk Maya to produce an interactive 3-dimensional image.

Ripser and GUDHI are used to compute and visulize persistent homology of the the Plaquette Eden Model. If you use this functionality, make sure to cite this library.

To represent the topology and local geometry of the Eden growth model, the software can build plots showing the following:
* the frequencies of the changes in Betti numbers (Figure 29 in the thesis),
* the distribution of volumes of top dimensional "holes" (the finite components of the complement of the structure due to Alexander duality, Figure 33 in the paper),
* the frequencies of top dimensional holes with specific shapes with 3 and 4 cells (Figure 32 in the thesis),
* the growth of the Betti numbers and the perimeter (Figures 26 and 28 in the thesis),
* persistent homology barcode (Figure 37 in the thesis).
 
All plots and data files are saved in the project folder.

## Acknowledgments
Erika Roldan was supported in part by NSF-DMS #1352386 and NSF-DMS #1812028 during 2018-2019. <br />
This project received funding from the European Union’s Horizon 2020 research and innovation program under the
Marie Skłodowska-Curie grant agreement No. 754462.

## Citations 

If you use the computations and visualization of persistent homology, cite the Ripser and the GUDHI packages.

## Dependencies:

Python 3.8.

Ripser. https://github.com/Ripser/ripser
GUDHI. http://gudhi.gforge.inria.fr/

## Ripser Installation

Here is how to obtain, build, and run Ripser:
```
git clone https://github.com/Ripser/ripser.git
cd ripser
make
./ripser examples/sphere_3_192.lower_distance_matrix
```

## Gudhi Installation

To install this package with conda run one of the following:
```
conda install -c conda-forge gudhi
```
```
conda install -c conda-forge/label/cf201901 gudhi
```
```
conda install -c conda-forge/label/cf202003 gudhi
```

## Usage
Installation and usage are done through the Command Line Interface. It is expected that a Unix-like system is used, such as Mac OS X or any Linux flavor. For Windows users, we advise looking into Windows Subsystem for Linux (WSL). On the operational system, Python 3 and Git should be installed. Run the following commands to obtain the Plaquette Eden Model package:
```
git clone https://github.com/annakrymova/Homology_and_Geometry_of_the_Plaquette_Eden_Model.git
cd ./Homology_and_Geometry_of_the_Plaquette_Eden_Model
```
Then, we suggest to create a virtual environment and install all required libraries, using the following commands:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
To launch the application, use the following command:
```
python3 ./main.py
```

At first, the system will ask you to choose the type of model
```
Specify the model type: 
    0 -- plaquette eden model 
    1 -- euler-characteristic mediated plaquette eden model
    2 -- filled-in cubes plaquette eden model
```

At the next step, you decide if you want a picture:
```
Do you want a picture of the model? (for a large model, this may take a long time) 
    0 -- no 
    1 -- yes
```
If a positive answer to the previous question was given, then you chooses the way to visualize the model:
```
Do you want Python or MAYA 3D model? (we would not recommend Python for models of size more than 500 cells). 
    0 -- Python 
    1 -- MAYA
```
Then, the system asks you to enter the size of the model:
```
How many cells would you like in your model?
```
And then you have to specify the number of models you want to generate:
```
How many models would you like to build?
```

After that, the modeling and analysis take place.  

When all calculations are finished, you will see the sentence:
```
WE ARE DONE! CHECK THE FOLDER!
```
Now, you can check the results in the corresponding folder. The results of the obtained models are saved in the folders of the format *#cells_date_time_model* inside the *experiments* folder. For example, if one simulated two Filled-In Cubes Plaquette Eden Models with 10,000 cells each, possible two folders that software will generate are:
```
    10k_09.05.2021_14.32.080_2
    10k_09.05.2021_14.32.471_2.
```




