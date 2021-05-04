## Homology-and-Geometry-of-the-Plaquette-Eden-Model


DEVELOPERS: <br />
Anna Krymova (anna.krymova@tum.de) <br />
Erika Roldan (erika.roldan@ma.tum.de) <br />


DATE: MAY 15, 2021

LICENSE: GNU GENERAL PUBLIC LICENSE (see license.txt)

## Overview 
This package includes software to run simulations of the Plaquette Eden Model, which is a discrete 2-dimensional stochastic cell growth process defined on the 3-dimensional regular cubical tessellation of the Euclidian space.  The sofrware analyzes the topology (Betti numbers and persistent homology) and local geometry of the structure. 
For graphical representation, the program can create a picture of the Plaquette Eden Model, or alternitevly can output a .txt file which can be inputed to MAYA to produce an interactive 3-dimensional image.

Ripser and GUDHI are used to compute and visulize persistent homology of the the Plaquette Eden Model. If you use this functionality, make sure to cite this library.

To represent the topology and local geometry of the Eden growth model, the software can build plots showing the following:
* the frequencies of the changes in Betti numbers (Figure 6.1 in the paper),
* the distribution of volumes of top dimensional "holes" (the finite components of the complement of the structure due to Alexander duality, Figure 6.5 in the paper),
* the growth of the Betti numbers and the perimeter (Figure 6.2 in the paper),
* in two and three dimensions, the frequencies of top dimensional holes with specific shapes with 3 and 4 cells (Table 6.4 in the paper) 

The folder *2d/files* contains the data of the 2D simulations that were used in the paper TOPOLOGY AND LOCAL GEOMETRY OF THE EDEN MODEL https://arxiv.org/pdf/2005.12349.pdf.

All plots and graphs are saved in the project folder.

## Acknowledgments
Erika Roldan was supported in part by NSF-DMS #1352386 and NSF-DMS #1812028 during 2018-2019. <br />
This project received funding from the European Union’s Horizon 2020 research and innovation program under the
Marie Skłodowska-Curie grant agreement No. 754462.

## Citations 

If you use this code, cite the paper TOPOLOGY AND LOCAL GEOMETRY OF THE EDEN MODEL. 
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
Download the whole folder and run the file *main.py*.<br />
At first, the system will ask you to enter the dimension:
```
Please, enter the desired dimension of your model (from 2 to 5): 
```
Then, you have to specify if you have a file with a pre-defined model or not:
```
Do you have a file with a model? 
0 -- you want to generate a new model 
1 -- you have a file
```
In the case of а 2d or а 3d model, on the next step you decide if you want a picture:
```
Do you want a picture of your model? (with a large model it can take time)  
0 -- no 
1 -- yes
```
In case you want to read the cubical complex from a file, the program asks you to specify the file format:
```
What is the format of the file? 
0 -- list of tuples 
1 -- Perseus
```
And then you should give a name of a file. Important: the file should be in the folder *files* inside the folder with the corresponding dimension, e.g. *5d/files*.
```
Name of the file (for example, filename.txt):
```
If you chose to generate a new model, then now the system asks you to enter the size of the model:
```
How many tiles would you like in your model?
```
And then you have to specify the number of models you want to generate:
```
How many models would you like to build?
```
In the end, the program asks you if you want to generate GUDHI barcodes:
```
Do you want GUDHI barcode(s)? 
0 -- no 
1 -- yes
```
and if yes, it asks you to specify the minimum length of the interval for every barcode (depending on the dimension of the model).
It is done in order to omit short-lived homology groups, i.e. short intervals.
<br />
After that, the modeling and analysis take place.  
<br />
When all calculations are finished, you will see the sentence:
```
WE ARE DONE! CHECK THE FOLDER!
```
Now, you are welcome to check the results in the corresponding folder. 
<br />
In case you have analyzed a model from a file, the program creates a folder that has the name of the file inside the folder with the corresponding dimension.
For example, if you analyze a 3-dimension model from a file *data.txt*, then the analysis results can be found in the folder:
```
3d/data.txt
```
Otherwise, that is, if you generate your own models. The results of the obtained models are saved in the folders *#tiles_date_time* inside the folder with the corresponding dimension.
For example, if you generate two 4-dimensional models with 10 000 tiles each, possible two folders that software will generate are:
```
4d/10k_11/12/2020_15.07.04
```
```
4d/10k_11/12/2020_15.07.21
```




