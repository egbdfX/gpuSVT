# gpuSVT

**gpuSVT** is a GPU accelerated singular value thresholding software written in Python.

## Installation

On Linux, after downloading the software package from https://github.com/egbdfX/gpuSVT, the programme can be built with:
```
conda create --name SVTexample python=3.9
conda activate SVTexample
cd /path/to/code
make
pip install cupy-cuda
pip install scipy
pip install mat4py
```
The main function of the package is in *gpuSVT_main.py*.

## Example

Download the *example* file, which includes three input files (*ini.mat*, *pyinp.mat*, *pymask.mat*) and a script file (*gpuSVT_example.py*).

Run the script by:
```
python gpuSVT_example.py
```
The output files with names start by *"nnR"* are the complete matrices. The output files with names start by *"recon_error"* are the reconstruction errors. Multiple complete matrices and their corresponding reconstruction errors will be outputted separately.

## Reference

When referencing this code, please cite our related paper:

X. Li, K. Adamek, W. Armour, “A GPU accelerated singular value thresholding software written in Python”.
