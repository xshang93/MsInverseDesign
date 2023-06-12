# Tailoring the mechanical properties of 3D microstructures: a deep learning and genetic algorithm inverse optimization framework

Xiao Shang

email: xiao.shang@mail.utoronto.ca

![TheFrmework](https://github.com/xshang93/MsInverseDesign/blob/main/abstract.jpg)

We report a framework that provides an end-to-end solution to achieve application-specific mechanical properties by microstructure optimization. In this study, we select the widely used Ti-6Al-4V to demonstrate the effectiveness of this framework by tailoring its microstructure and achieving various yield strength and elastic modulus across a large design space, while minimizing the stress concentration factor. Compared with conventional methods, our framework is efficient, versatile, and readily transferrable to other materials and properties.

### Core dependencies and librarys

- MATLAB python engine R2023a
- Neper 4.4.2 - 33 (https://neper.info/) A free / open source software package for polycrystal generation and meshing.
- Numpy 1.23.5 
- Tensorflow 2.10.1
- Pandas 1.5.3
- PyGAD 2.18.1 (https://pygad.readthedocs.io/en/latest/) An open-source Python library for building the genetic algorithm.
- python 3.9.16
- Scikeras 0.9.0
- Sklearn 1.0.2

### How to use
1. Make sure your have all required dependencies at the correct versions.
2. Clone this repo to your local directory uisng ```git clone https://github.com/xshang93/MsInverseDesign/```
3. In the ```GA_main_final.py``` file, line 204, select the ojb_select to be either 4 or 5. 4 is for maximizing both strength and elastic modulus, and 5 is for maximizing stength but minimizing modulus. Both minimizes the stress concentration factor.
4. Wait for the optimization loop to converge. The results and history will be stored in a folder named ```opt_run```.
