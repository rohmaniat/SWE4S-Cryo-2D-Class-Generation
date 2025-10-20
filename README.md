# Overview

This software is a neural network that is designed to be able to pick particles from Cryo-EM micrographs. This is an important step in the Cryo data pipeline. We will be training and testing our model on an excellent data set found in a Nature article.

The paper providing the training and testing data can be found here: <https://www.nature.com/articles/s41597-023-02280-2>

Their GitHub containing the data is here: <https://github.com/BioinfoMachineLearning/cryoppp>

A group created a software using CryoPPP to train a neural network for particle picking (basically exactly what we are doing). Here's their GitHub: <https://github.com/jianlin-cheng/CryoTransformer/tree/main>

## File Descriptions

The class_def.py file provides the class definition for the Dataset object.

The train.py file is meant for (you guessed it!) training!

The utils.py file is for utilities. It will contain functions like train() and test(). We could also put the NN class def in here if we want.

**These files/folders might no longer be needed:**

The particle_picker.py file is where we will build, train, and test the network. It is designed to read the .mrc image files and .csv files and incorporating them into either training or testing data sets.

The PyTorch Tutorial folder contains a great tutorial for how to use PyTorch. The website online also has some useful commentary about how it works.

The controller.sh file should sort of be our control panel. We won't want to download the MASSIVE training data onto our local machines, so we'll somehow need to make this file call in that data onto the supercomputer. This file will also probably operate the particle_picker.py file (argparse, anyone?).

## BUILDING THE ENVIRONMENT

`mamba env create -f particle_env.yml`
`mamba activate particle`
This is how we will recreate the environment in the HPC

How to create a new environment (assuming that it's already active):
`mamba env export > particle_env.yml`

Don't worry about these commands for now- I need them to remember how to set up an environment
conda create -n particle
mamba activate particle
conda activate my-env
conda env export > environment.yml

## TODO

- Give more specificity to the particle_env.yml file. We want to make sure anyone can run it.
- Make a bunch of unit and functional tests. Let's make sure we really catch all our edge cases.

## Notes

- In the PyTorch tutorial, they use the nn.CrossEntropyLoss() function. This is a good loss function for classifying images. Since we'll first need to do particle picking, I think we should start with a loss function like nn.MSELoss() or nn.L1Loss() (which does mean squared error or absolute value error for regression objectives like particle picking). Once we've got that going and once we can do the image slicing step, then we could start looking at the nn.CrossEntropyLoss() function.
- ALL FILES expect you to run them from the root directory (you should type bash src/controller.sh to execute the controller.sh file)
- Some of the MRC files have bad headers. The permissive=True flag when opening the mrc file should be able to get past this. I've wrapped it with a warnings catch so we aren't bothered with those.
- The folder layout looks like this:

``` plaintext
JLRparticle/
├── SWE4S-Cryo-2D-Class-Generation/    (This is the root directory from where we will run our scripts)
│   ├── README.md
│   ├── src/
│   │   ├── utils.py
│   │   ├── controller.sh
│   │   ├── particle_env.yml
│   │   ├── particle_picker.py
│   │   └── ...
│   ├── test/
│   │   ├── func/
│   │   │   └── func_testing.sh
│   │   └── unit/
└── Data/  (NOT in repo)
```
