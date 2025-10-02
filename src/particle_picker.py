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
import sys
import argparse
import os

sys.path.append("src/") # noqa

import utils


if __name__ == '__main__':

    # Build the argument parser
    parser = argparse.ArgumentParser(description="Image and CSV data extraction")
    parser.add_argument("enzyme_code", type=int, help="This is the numeric label of the enzyme of interest (ex: 10017)")
    parser.add_argument("-s", "--split", type=bool, default=False, help="'True' here will command that the extracted data be split into training and testing data")

    # Parse the arguments
    args = parser.parse_args()
    enzyme_num = args.enzyme_code

    '''
    # this is the core of how we will handle file paths in the supercomputer

    file_path = os.path.join("sample_ground_truth","")
    csv_file_path = os.path.join("sample_micrographs","")
    '''

    # may need to add ".." to the arguments if we put this in a subfolder

    # Provide name of the MRC file and its corresponding CSV file
    # Note that these lines will need to be replaced to be able to handle huge data

    # Relative file paths (TODO: Build a file-navigator that will go through files and feed them into the NN)
    #img_file_path = r"..\Data\10017\micrographs\Falcon_2012_06_12-14_33_35_0.mrc"
    #csv_file_path = r"..\Data\10017\ground_truth\particle_coordinates\Falcon_2012_06_12-14_33_35_0.csv"

    # Jared's absolute file paths
    #img_file_path = r"C:\Users\jared\Desktop\Schoolwork\Software Engineering for Scientists\Semester Project\Data\10017\micrographs\Falcon_2012_06_12-14_33_35_0.mrc"
    #csv_file_path = r"C:\Users\jared\Desktop\Schoolwork\Software Engineering for Scientists\Semester Project\Data\10017\ground_truth\particle_coordinates\Falcon_2012_06_12-14_33_35_0.csv"

    #Rohan's absolute file paths
    #img_file_path = r"/Users/rohan/local_coding/10005/micrographs//Users/rohan/local_coding/10005/micrographs/stack_0002_2x_SumCorr.mrc"
    #csv_file_path = r"/Users/rohan/local_coding/10005/ground_truth/particle_coordinates/stack_0002_2x_SumCorr.csv"


    # Collect file names
    images, csvs = utils.data_extractor(enzyme_num)

    # Build file paths
    img_file_path = "../Data/" + str(enzyme_num) + "/micrographs/" + images[0]          # I'm just starting with the first one to make sure everything is working
    csv_file_path = "../Data/" + str(enzyme_num) + "/ground_truth/particle_coordinates/" + csvs[0]

    print(f"First image file path: {img_file_path}")
    print(f"First CSV file path: {csv_file_path}")

    # Open the mrc file
    try:
        with mrcfile.open(img_file_path) as mrc:
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
        print(f"Error: The file '{img_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while opening the MRC file: {e}")
        sys.exit(2)

    # Open the csv file and read in the particles' coordinates and diameter into a PyTorch tensor
    try:
        particle_coords = pd.read_csv(csv_file_path)
        selected_data = particle_coords[['X-Coordinate', 'Y-Coordinate', 'Diameter']]
        particles = torch.tensor(selected_data.values).float()

        print(f"Shape of the coords tensor: {particles.shape}")          # Output: Shape of the coords tensor: torch.Size([629, 3])
        print(f"Data type of the coords tensor: {particles.dtype}")      # Output: Data type of the coords tensor: torch.float32

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while working with the CSV file: {e}")
        sys.exit(2)