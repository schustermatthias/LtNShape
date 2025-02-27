This is the code for obtaining the results of the paper *Interface Identification Constrained by Local-to-Nonlocal Coupling * by M. Schuster and V. Schulz that can be found on https://arxiv.org/abs/2402.12871.

Build and Install on Ubuntu
===========================
In order to clone the project do
::
  git clone https://github.com/schustermatthias/LtNShape.git path/to/local_folder

| Since this code contains a customized version of **nlfem** the following **basic requirements** of nlfem are needed
| ``gcc, g++, python3-dev, python3-venv, libgmp-dev, libcgal-dev, metis, libmetis-dev, libarmadillo-dev``.
On Ubuntu, this can be done via
::
  sudo apt-get install git gcc g++ libarmadillo-dev liblapack-dev libmetis-dev
  sudo apt-get install python3-venv python3-dev libgmp-dev libcgal-dev

| Moreover, to run nlshape **legacy FEniCS(version 2019.1.0)** is required. In order to use FEniCS in a virtual environment, it may has to be installed globally and then inherited as a global site package. 
A virtual environment can be built and activated via
::
  mkdir venv
  python3 -m venv venv/
  source venv/bin/activate

Additionally, the packages from the file **requirements.txt** are neccessary and can be installed by
::
  (venv) python3 -m pip install -r requirements.txt

The creation of the virtual environment and the installation of packages from requirements.txt can probably also be done via your IDE.
Finally, nlfem can be installed by
::
  (venv) python3 setup.py build --force install

| See https://gitlab.uni-trier.de/pde-opt/nonlocal-models/nlfem for more information.
| Further, Gmsh needs to be installed to run the examples where the mesh is remeshed. On Ubuntu Gmsh can be installed via
::
   sudo apt install gmsh

More information on Gmsh can be found on https://gmsh.info.
Running the Examples from the Paper
===================================
To run one of the examples of the paper or the PhD thesis, choose in line 8 of main.py the associated configuration file, i.e., "configuration_ex1", "configuration_ex2", "configuration_ex3" or "configuration_ex4".  

Raw Data
========
The data of the experiments in the paper can be found in the folder "LtNShape/results".

License
=======
LtNShape is published under GNU General Public License version 3. Copyright (c) 2025 Matthias Schuster

| Parts of the project are taken from **nlfem** and have been customized.
| nlfem is published under GNU General Public License version 3. Copyright (c) 2021 Manuel Klar, Christian Vollmann
  
