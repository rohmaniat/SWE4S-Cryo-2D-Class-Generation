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

The train.py file is meant for (you guessed it!) training! It outputs a model.pt file.

The predict.py file uses a trained model to predict particle locations!

The class_def.py file provides the class definition for the Dataset object.

The utils.py file is for utilities. It has a few useful functions (including the find all data function).

There are separate folders in src/ for models, predictions, and images of the data.

There is a separate folder for slurm scripts. These are only needed for downloading the (massive) datasets to a compute cluster or training the model.

## Notes

- ALL FILES expect you to run them from the root directory (you should type bash src/controller.sh to execute the controller.sh file)
- Some of the MRC files have bad headers. The permissive=True flag when opening the mrc file should be able to get past this. I've wrapped it with a warnings catch so we aren't bothered with those.
- The folder layout looks like this:

``` plaintext
JLRparticle/
├── SWE4S-Cryo-2D-Class-Generation
|    ├── cryo-picker.yml
|    ├── particle_env.yml
|    ├── README.md
|    ├── src
|    │   ├── class_def.py
|    │   ├── config.ini
|    │   ├── images
|    │   │   └── (images)
|    │   ├── models
|    │   │   ├── (models)
|    │   ├── predict.py
|    │   ├── visualization
|    │   │   └── (visualizations)
|    │   ├── slurm
|    │   │   ├── pielcontroller.sh
|    │   │   ├── sample.sh
|    │   │   └── train.sh
|    │   ├── train.py
|    │   └── utils.py
|    ├── ssshtest
|    └── test
|        │   └── __init__.cpython-313.pyc
|        ├── func
|        │   ├── EXAMPLE_Falcon_2012_06_12-14_33_35_0.csv
|        │   ├── EXAMPLE_Falcon_2012_06_12-14_33_35_0.mrc
|        │   ├── func_testing.sh
|        │   └── standard_config.ini
|        └── unit
|            ├── epoch_training_tests.py
|            ├── fake_model.py
|            └── utils_tests.py
└── Data/  (NOT in repo)
```
