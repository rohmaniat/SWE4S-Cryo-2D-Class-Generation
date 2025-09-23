import torch

"""
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
"""

import mrcfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
# this is the core of how we will handle file paths in the supercomputer
import os

file_path = os.path.join("sample_ground_truth","")
csv_file_path = os.path.join("sample_micrographs","")
'''

# may need to add ".." to the arguments if we put this in a subfolder

# Provide name of the MRC file and its corresponding CSV file
# Note that these lines will need to be replaced to be able to handle huge data

# Jared's file paths
# file_path = r"C:\Users\jared\Desktop\Schoolwork\Software Engineering for Scientists\Semester Project\10017\micrographs\Falcon_2012_06_12-14_33_35_0.mrc"
# csv_file_path = r"C:\Users\jared\Desktop\Schoolwork\Software Engineering for Scientists\Semester Project\10017\ground_truth\particle_coordinates\Falcon_2012_06_12-14_33_35_0.csv"

#Rohan's file paths
file_path = r"/Users/rohan/local_coding/10005/micrographs//Users/rohan/local_coding/10005/micrographs/stack_0002_2x_SumCorr.mrc"
csv_file_path = r"/Users/rohan/local_coding/10005/ground_truth/particle_coordinates/stack_0002_2x_SumCorr.csv"

# TODO - make the file paths relative to the project directory

# First, open the mrc file
try:
    with mrcfile.open(file_path) as mrc:
        data = mrc.data         # Note we could do mrc.header but I don't think we need that info

        # Let's take a look at the actual image
        """
        plt.imshow(data, cmap='gray')
        plt.title("Cryo-EM Micrograph")
        plt.colorbar()
        plt.show()
        """

        # Convert the NumPy array into a PyTorch tensor
        image_tensor = torch.tensor(data).float()

        print(f"image_tensor shape: {image_tensor.shape}")
        print(f"image_tensor data type: {image_tensor.dtype}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred while opening the MRC file: {e}")

# Open the csv file and read in the particles' coordinates and diameter into a PyTorch tensor
try:
    particle_coords = pd.read_csv(csv_file_path)
    selected_data = particle_coords[['X-Coordinate', 'Y-Coordinate', 'Diameter']]
    particles = torch.tensor(selected_data.values).float()

    #print(f"Shape of the final tensor: {particles.shape}")          # Output: Shape of the final tensor: torch.Size([629, 3])
    #print(f"Data type of the final tensor: {particles.dtype}")      # Output: Data type of the final tensor: torch.float32

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred while working with the CSV file: {e}")