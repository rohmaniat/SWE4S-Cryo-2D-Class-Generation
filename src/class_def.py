import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import mrcfile
import sys

sys.path.append("src/")  # noqa

import utils


def collate(batch):
    """
    This is a collate function that will help with pre-processing.
    Before feeding into the NN, we need to combine the CSV data so that it
    doesn't have any problems with the fact that each micrograph will have
    a different number of particles in it.

    Note the input (batch) is a list of tuples containing micrograph data
    followed by coordinate data
    """

    # Separate the mircographs and coordinates from the batch
    micrographs = [item[0] for item in batch]
    coordinates = [item[1] for item in batch]

    # Stack the micrographs (since they're all the same size)
    stacked_micrographs = torch.stack(micrographs, 0)

    # Note we leave the coordinates alone (as a list of tensors)

    return stacked_micrographs, coordinates


class CryoEMDataset(Dataset):
    """
    Custom PyTorch Dataset for Cryo-EM Micrographs and particle coordinates
    """
    def __init__(self, mrc_paths, csv_paths, transform=None):
        # This is just to initialize the object
        self.mrc_paths = mrc_paths
        self.csv_paths = csv_paths
        self.transform = transform

    def __len__(self):
        # This is just the length of the object
        return len(self.mrc_paths)  # We could also call self.csv_paths

    def __getitem__(self, idx):
        # Get the full paths for the MRC file and the CSV file
        mrc_path = self.mrc_paths[idx]
        csv_path = self.csv_paths[idx]

        # Load the MRC data
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            micrograph = mrc.data.copy()
        # Apply transforms (if they exist)
        if self.transform:
            micrograph = self.transform(micrograph)

        # Load the CSV data
        coordinates = pd.read_csv(csv_path)
        coordinates = torch.from_numpy(coordinates.values)

        return micrograph, coordinates
