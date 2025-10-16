"""
This script is to contain utility functions like train() and test()
"""

import os
import sys


def pull_micrographs(enzyme_code):
    directory = '../Data/' + str(enzyme_code) + '/micrographs'
    micrograph_filecount = 0

    # looks in the micrographs directory for all files
    # prints all file names and returns the number of files
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            #print(filename)
            micrograph_filecount += 1

    return micrograph_filecount


def pull_coordinates(enzyme_code):
    directory = '../Data/' + str(enzyme_code)
    directory += '/ground_truth/particle_coordinates'
    coords_filecount = 0

    # looks in ground_truth/particle_coordinates
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            # print(filename)
            coords_filecount += 1

    return coords_filecount


def data_info(enzyme_code):
    enzyme_code = str(enzyme_code)
    # collects the filenames from specified folders and stores them as a list
    image_names = os.listdir("../Data/" + enzyme_code + "/micrographs/")
    path = "../Data/" + enzyme_code + "/ground_truth/particle_coordinates/"
    csv_names = os.listdir(path)

    # print(len(image_names))
    # print(len(csv_names))
    return image_names, csv_names


def data_extractor(enzyme_code):
    """
    The purpose of this function is to look at a single enzyme file and return
    a tuple of two arrays: one containing the MRC file names and the other
    containing the CSV file names.

    This function also finds any discrepencies between the number of MRC and
    CSV files. If there are more MRC files, it will delete all the ones that
    don't have a corresponding CSV file.
    """

    # Gather ALL the files in the correct directory
    image_dir = "../Data/" + str(enzyme_code) + "/micrographs/"
    all_image_files = os.listdir(image_dir)
    csv_dir = "../Data/" + str(enzyme_code) + "/ground_truth/particle_coordinates/"
    all_csv_files = os.listdir(csv_dir)

    # Only extract the mrc and csv files
    image_names = [f for f in all_image_files if f.endswith(".mrc")]
    csv_names = [f for f in all_csv_files if f.endswith(".csv")]

    # Sort (for good practice)
    image_names.sort()
    csv_names.sort()

    # Make sure the lists are of equal length
    if len(image_names) == len(csv_names):
        combined = (image_names, csv_names)
        return combined

    # If they aren't, there's probably more mrc files than csv files
    elif (len(image_names) > len(csv_names)):
        print(f"Enzyme {enzyme_code} has a mismatch. ",
              "Filtering micrographs now.")

        # We need to weed out the mismatching files.
        # First, remove the ".csv" and ".mrc" from each file name
        image_names = [s.replace(".mrc", "") for s in image_names]
        csv_names = [s.replace(".csv", "") for s in csv_names]

        # Next, find the names that appear in both lists
        common_files = [name for name in image_names if name in csv_names]

        # Now rebuild lists
        image_names = [f"{file_name}.mrc" for file_name in common_files]
        csv_names = [f"{file_name}.csv" for file_name in common_files]

        # Check to make sure there are now the same number of files
        if len(image_names) != len(csv_names):
            print("There is STILL a different number of MRC and CSV files!",
                  " That's bad!")

        combined = (image_names, csv_names)
        return combined

    # If there are more csv files than mrc files, I dunno what
    # the heck is going on
    else:
        print(f"Take a look at enzyme {enzyme_code}. There are fewer MRC ",
              "files than CSV files. That's weird!")
        sys.exit(3)


def find_all_data():
    """
    This function is meant to look through the Data/ folder to find all the
    filenames available.

    It will return two lists.
    The first list is all the MRC files (the entire filepath).
    The second list is all the CSV files (again the entire filepath)
    """
    all_MRC = []
    all_CSV = []
    MRC_INDEX = 0
    CSV_INDEX = 1

    enzymes_available = os.listdir("../Data/")

    # Run through for each available enzyme
    for enzyme in enzymes_available:
        file_names = data_extractor(enzyme)

        # Build the full MRC filepaths
        for i in range(len(file_names[MRC_INDEX])):
            full_path = os.getcwd() + "/../Data/" + enzyme + "/micrographs/"
            full_path += file_names[MRC_INDEX][i]
            all_MRC.append(full_path)
        
        # Build the full CSV filepaths
        for i in range(len(file_names[CSV_INDEX])):
            full_path = os.getcwd() + "/../Data/" + enzyme
            full_path += "/ground_truth/particle_coordinates/"
            full_path += file_names[CSV_INDEX][i]
            all_CSV.append(full_path)

    return all_MRC, all_CSV


if __name__ == "__main__":
    thing = find_all_data()
    print(thing[1][4])
