# README

The paper providing the training and testing data can be found here: https://www.nature.com/articles/s41597-023-02280-2
Their GitHub containing the data is here: https://github.com/BioinfoMachineLearning/cryoppp

The PyTorch Tutorial folder contains a great tutorial for how to use PyTorch. The website online also has some useful commentary about how it works.

The particle_picker.py file is where we will build, train, and test the network.

The utils.py file is for utilities. It will contain functions like train() and test(). We could also put the NN class def in here if we want.

The controller.sh file should sort of be our control panel. We won't want to download the MASSIVE training data onto our local machines, so we'll somehow need to make this file call in that data onto the supercomputer. This file will also probably operate the particle_picker.py file (argparse, anyone?).

TODO:
- Look at the repo from that paper and see how they've provided the data. We'll need to figure out what pre-processing we'll need to do before feeding it into our NN.
- Other stuff

Notes:
- In the PyTorch tutorial, they use the nn.CrossEntropyLoss() function. This is a good loss function for classifying images. Since we'll first need to do particle picking, I think we should start with a loss function like nn.MSELoss() or nn.L1Loss() (which does mean squared error or absolute value error for regression objectives like particle picking). Once we've got that going and once we can do the image slicing step, then we could start looking at the nn.CrossEntropyLoss() function.
- Y'all are great. Don't forget it
