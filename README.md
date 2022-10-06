# 3D Generative Model Latent Disentanglement via Local Eigenprojection

## Installation

After cloning the repo open a terminal and go to the project directory. 

Change the permissions of install_env.sh by running `chmod +x ./install_env.sh` 
and run it with:
```shell script
./install_env.sh
```
This will create a virtual environment with all the necessary libraries.

Note that it was tested with Python 3.8, CUDA 11.3, and Pytorch 1.12.1. The code 
should work also with newer versions of  Python, CUDA, and Pytorch. If you wish 
to try running the code with more recent versions of these libraries, change the 
CUDA, TORCH, TORCHVISION, TORCH_GEOM_VERSION, and PYTHON_V variables in 
install_env.sh

Then activate the virtual environment :
```shell script
source ./id-generator-env/bin/activate
```


## Datasets

To obtain access to the UHM models and generate the dataset, please follow the 
instructions on the 
[github repo of UHM](https://github.com/steliosploumpis/Universal_Head_3DMM).

 Data will be automatically generated from the UHM during the first training. 
 In this case the training must be launched with the argument `--generate_data` 
 (see below).
 
 Alternatively, download the [STAR model](https://star.is.tue.mpg.de/) to 
 generate body shapes or the [LYHM](https://www-users.cs.york.ac.uk/~nep/research/LYHM/)
 and [CoMA](https://coma.is.tue.mpg.de/) to train the network on 
 data from real subjects.
 
 ## Prepare Your Configuration File
 
 We made available a configuration file for each experiment (default.yaml is 
 the configuration file of LED-VAE). Make sure that
 the paths in the config file are correct. In particular, if you are generating 
 data from a pca model, you might have to change `pca_path` according to the
 location where UHM/STAR was downloaded.
 
 ## Train and Test
 
 To start the training from the project repo simply run:
 ```shell script
python train.py --config=configurations/<A_CONFIG_FILE>.yaml --id=<NAME_OF_YOUR_EXPERIMENT>
```

If this is your first training and you wish to generate the data, run:
```shell script
python train.py --generate_data --config=configurations/<A_CONFIG_FILE>.yaml --id=<NAME_OF_YOUR_EXPERIMENT>
``` 

Basic tests will automatically run at the end of the training. If you wish to 
run additional tests presented in the paper you can uncomment any function call 
at the end of `test.py`. If your model has already been trained or you are using 
our pretrained models, you can run tests without training:

```shell script
python test.py --id=<NAME_OF_YOUR_EXPERIMENT>
```
Note that NAME_OF_YOUR_EXPERIMENT is also the name of the folder containing the
pretrained model.


## Demo notebook
`demo.ipynb` contains a demo that even people without access to the original 
dataset can use. 
The demo will allow you to randomly generate head models and then edit each 
attribute by manually changing the values of the latent variables.

To run the demo download the `demo_files` folder and copy it in the project 
directory. The demo files within the `demo_files` folder are:
 - the precomputed down- and up-sampling transformations,
 - the precomputed spirals,
 - the mesh template segmented according to the different shape attributes,
 - the network weights for different models,
 - the data normalisation values.
 
 
 ## SD-VAE
 This code is derived from the SD-VAE, which is still available for sake of 
 comparison. The SD-VAE model can be trained, tested, and demoed with the
 code in this repository. If you use SD-VAE, please cite the original paper.
 
 We recommend you to consider both models when developing your specific 
 application because each of them comes with its own strengths. For instance,
 SD-VAE may be more easily applicable to other data representations and it
 generates more symmetric shapes when using the attribute segmentation proposed 
 in the paper. Please refer to the 
 **3D Generative Model Latent Disentanglement via Local Eigenprojection** paper
 to find out the LED models improvements over SD-VAE.
 