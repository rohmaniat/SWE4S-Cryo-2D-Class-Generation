"""
This script is to contain utility functions like train() and test()
"""

import os
import sys

def pull_micrographs(enzyme_code):
    directory = '../Data/' + str(enzyme_code) + '/micrographs'
    micrograph_filecount = 0
    print(directory) # prints the filepath

    # looks in the micrographs directory for all files
    # prints all file names and returns the number of files
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            print(filename)
            micrograph_filecount += 1
    print(micrograph_filecount)
    return micrograph_filecount


def pull_coordinates(enzyme_code):
    directory = '../Data/' + str(enzyme_code) + '/ground_truth/particle_coordinates'
    coords_filecount = 0
    print(directory) # prints the filepath

    # looks in ground_truth/particle_coordinates
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            print(filename)
            coords_filecount += 1
    print(coords_filecount)
    return coords_filecount


def data_info(enzyme_code):
    # collects the filenames from the specified folders and stores them as a list
    image_names = os.listdir("../Data/" + str(enzyme_code) + "/micrographs/")
    csv_names = os.listdir("../Data/" + str(enzyme_code) + "/ground_truth/particle_coordinates/")

    print(len(image_names))
    print(len(csv_names))


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


if __name__ == "__main__":
    pull_micrographs(10005)