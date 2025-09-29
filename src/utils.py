"""
This script is to contain utility functions like train() and test()
"""

import os
import sys

def data_extractor(enzyme_code):
    """
    The purpose of this function is to look at a single enzyme file and return two arrays: one containing the MRC file names and the other containing the CSV file names.
    """
    
    image_names = os.listdir("../Data/" + str(enzyme_code) + "/micrographs/")
    csv_names = os.listdir("../Data/" + str(enzyme_code) + "/ground_truth/particle_coordinates/")

    # Make sure the lists are of equal length
    if len(image_names) == len(csv_names):
        combined = (image_names,csv_names)
        return combined
    else:
        print(f"Take a look at enzyme {enzyme_code}. There are a different number of MRC and CSV files.")
        sys.exit(3)