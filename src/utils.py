"""
This script is to contain utility functions like train() and test()
"""

import os
import sys

def pull_micrographs(enzyme_code):
    directory = '../Data/' + str(enzyme_code) + '/micrographs'
    micrograph_filecount = 0
    #print(directory) # prints the filepath

    # looks in the micrographs directory for all files
    # prints all file names and returns the number of files
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            #print(filename)
            micrograph_filecount += 1
    #print(micrograph_filecount)
    return micrograph_filecount


def pull_coordinates(enzyme_code):
    directory = '../Data/' + str(enzyme_code) + '/ground_truth/particle_coordinates'
    coords_filecount = 0
    #print(directory) # prints the filepath

    # looks in ground_truth/particle_coordinates
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            #print(filename)
            coords_filecount += 1
    #print(coords_filecount)
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
    This function also finds any discrepencies between the number of MRC and CSV files. If there are more MRC files, it will delete all the ones that don't have a corresponding CSV file.
    """
    
    # Gather ALL the files in the correct directory
    all_image_files = os.listdir("../Data/" + str(enzyme_code) + "/micrographs/")
    all_csv_files = os.listdir("../Data/" + str(enzyme_code) + "/ground_truth/particle_coordinates/")

    # Only extract the mrc and csv files
    image_names = [f for f in all_image_files if f.endswith(".mrc")]
    csv_names = [f for f in all_csv_files if f.endswith(".csv")]

    # Sort (for good practice)
    image_names.sort()
    csv_names.sort()

    # Make sure the lists are of equal length
    if len(image_names) == len(csv_names):
        combined = (image_names,csv_names)
        return combined
    
    # If they aren't, there's probably more mrc files than csv files
    elif (len(image_names) > len(csv_names)):
        print(f"Enzyme {enzyme_code} has a mismatch. Filtering micrographs now.")

        # We need to weed out the mismatching files.
        # First, remove the ".csv" and ".mrc" from each file name
        image_names = [s.replace(".mrc", "") for s in image_names]
        csv_names = [s.replace(".csv", "") for s in csv_names]

        # Next, find the names that appear in both lists
        common_files = [file_name for file_name in image_names if file_name in csv_names]
        
        # Now rebuild lists
        image_names = [f"{file_name}.mrc" for file_name in common_files]
        csv_names = [f"{file_name}.csv" for file_name in common_files]

        # Check to make sure there are now the same number of files
        if len(image_names) != len(csv_names):
            print("There is STILL a different number of MRC and CSV files! That's bad!")

        combined = (image_names,csv_names)
        return combined

    # If there are more csv files than mrc files, I dunno what the heck is going on
    else:
        print(f"Take a look at enzyme {enzyme_code}. There are a different number of MRC and CSV files.")
        sys.exit(3)

        print(f"Take a look at enzyme {enzyme_code}. There are fewer MRC files than CSV files. That's weird!")
        sys.exit(3)
        
       
if __name__ == "__main__":
    pull_micrographs(10005)
# There may be indentation problems in the last few lines here SORRY
