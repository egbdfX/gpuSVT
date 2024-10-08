# gpuSVT

**gpuSVT** is a GPU accelerated singular value thresholding software written in Python.

## Installation

On Linux, after downloading the software package from https://github.com/egbdfX/gpuSVT, the programme can be built with:
```
conda create --name SVTexample python=3.9
conda activate SVTexample
cd /path/to/src
make
pip install cupy-cuda (e.g., cupy-cuda12x)
pip install scipy
pip install mat4py
```
The main function of the package is in *gpuSVT_main.py*.

## Example

Download the *example* file, which includes three input files (*ini.mat*, *pyinp.mat*, *pymask.mat*) and a script file (*gpuSVT_example.py*). To run the example script, the three input files and the script file need to be put in the same directory of /src.

Please note that the images used in the inputs of the example are generated based on [File:Russell Falls 2.jpg](https://commons.wikimedia.org/wiki/File:Russell_Falls_2.jpg) which is licensed under the [Creative Commons Attribution-Share Alike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/deed.en) license, and [File:Galu Beach shore.jpg](https://commons.wikimedia.org/wiki/File:Galu_Beach_shore.jpg) which is licensed under the [Creative Commons Attribution-Share Alike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/deed.en) license. The input incomplete images are generated by greying, cropping and adding in salt-and-pepper noise on the original images.

Run the script by:
```
python gpuSVT_example.py
```
The output files with the names prefixed by *"nnR"* are the complete matrices. The output files with the names prefixed by *"recon_error"* are the reconstruction errors. Multiple complete matrices and their corresponding reconstruction errors will be outputted separately.

## Contact
If you have any questions or need further assistance, please feel free to contact at [egbdfmusic1@gmail.com](mailto:egbdfmusic1@gmail.com).

## Reference

When referencing this code, please cite our related paper:

X. Li, K. Adámek, W. Armour, "GPU accelerated singular value thresholding," SoftwareX, Volume 23, 2023, 101500.
https://doi.org/10.1016/j.softx.2023.101500

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
