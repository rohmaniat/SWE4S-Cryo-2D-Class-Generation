# Overview

This software, JLR Particle, is a neural network that is designed to be able to pick particles from Cryo-EM micrographs. This is an important step in the Cryo data pipeline, as the quality of generated 2D classes is foundational for creating a high-resolution final structure.

We will be training and testing our model on an excellent data set found in a Nature article. The paper providing the training and testing data can be found here: <https://www.nature.com/articles/s41597-023-02280-2> Their GitHub containing the data is here: <https://github.com/BioinfoMachineLearning/cryoppp>

A group created a software using CryoPPP to train a neural network for particle picking (basically exactly what we are doing). Here's their GitHub: <https://github.com/jianlin-cheng/CryoTransformer/tree/main>

## Getting Started

Training this software requires two different inputs -- micrograph files (.mrc) which are raw images from the microscope that does or does not contain particles of interest, and coordinate files (.csv) that contain the quantity and location of all particles from the ground truth sample.

As a user, you would need to input .mrc files from microscope data as well as a configuration file (.ini) and a trained model (.pt). We have provided one of each in this repository. There is a specific file structure that is required for the training and testing code to read files properly. Outside of this repo, create a directory called Data/ with all micrograph data (organized into subdirectories called micrographs/ and ground_truth/particle_coordinates/).

For those who want to test out this software, we have created an automated system using Snakemake! Run using the following code:

``` snakemake --cores 1 --use-conda ```

This file will use three tester .mrc files provided in sample_data and run a particle prediction using predict.py. The outputs will be stored in the "visualizations" folder.

## File Descriptions

The class_def.py file provides the class definition for the Dataset object.

The train.py file is meant for (you guessed it!) training!

The utils.py file is for utilities. It will contain functions like train() and test(). We could also put the NN class def in here if we want.

The predict.py file uses a trained model to predict particle locations!

There are separate folders in /src for models, predictions, and images of the data.

There is a separate folder for slurm scripts. These are only needed for downloading the (massive) datasets to a compute cluster or training the model.

The particle_picker.py file is where we will build, train, and test the network. It is designed to read the .mrc image files and .csv files and incorporating them into either training or testing data sets.

The PyTorch Tutorial folder contains a great tutorial for how to use PyTorch. The website online also has some useful commentary about how it works.

## ENVIRONMENT

This package requires a few key dependencies, namely PyTorch, Pandas, and MRCFile. The environment is called "particle" and can be set up using the following commands:

`mamba env create -f particle_env.yml`

`mamba activate particle`

## Notes

- In the PyTorch tutorial, they use the nn.CrossEntropyLoss() function. This is a good loss function for classifying images. Since we'll first need to do particle picking, I think we should start with a loss function like nn.MSELoss() or nn.L1Loss() (which does mean squared error or absolute value error for regression objectives like particle picking). Once we've got that going and once we can do the image slicing step, then we could start looking at the nn.CrossEntropyLoss() function.
- ALL FILES expect you to run them from the root directory (you should type bash src/controller.sh to execute the controller.sh file)
- Some of the MRC files have bad headers. The permissive=True flag when opening the mrc file should be able to get past this. I've wrapped it with a warnings catch so we aren't bothered with those.
- The folder layout looks like this:

``` plaintext
JLRparticle/
├── cryo-picker.yml
├── particle_env.yml
├── README.md
├── src
│   ├── class_def.py
│   ├── config.ini
│   ├── images
│   │   └── (images)
│   ├── models
│   │   ├── (models)
│   ├── predict.py
│   ├── visualization
│   │   └── (visualizations)
│   ├── slurm
│   │   ├── pielcontroller.sh
│   │   ├── sample.sh
│   │   └── train.sh
│   ├── train.py
│   └── utils.py
├── ssshtest
└── test
    │   └── __init__.cpython-313.pyc
    ├── func
    │   ├── EXAMPLE_Falcon_2012_06_12-14_33_35_0.csv
    │   ├── EXAMPLE_Falcon_2012_06_12-14_33_35_0.mrc
    │   ├── func_testing.sh
    │   └── standard_config.ini
    └── unit
        ├── epoch_training_tests.py
        ├── fake_model.py
        └── utils_tests.py
└── Data/  (NOT in repo)
```
