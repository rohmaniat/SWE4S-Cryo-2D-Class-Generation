import os
import sys
import torch


def pull_micrographs(enzyme_code):
    directory = '../Data/' + str(enzyme_code) + '/micrographs'
    micrograph_filecount = 0

    # looks in the micrographs directory for all files
    # prints all file names and returns the number of files
    try:
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                # print(filename)
                micrograph_filecount += 1
    except FileNotFoundError:
        print("The enzyme code could not be accessed.")
        sys.exit(2)

    if not micrograph_filecount:
        print("There were no micrograph files found.")
        return ValueError

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
    csv_dir = "../Data/" + str(enzyme_code)
    csv_dir += "/ground_truth/particle_coordinates/"

    # there are some data files with no csv files? (10075)
    if csv_dir:
        all_csv_files = os.listdir(csv_dir)
    else:
        return None

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
    # TODO: I have it return none so the train ignores those files, but I
    # think I can do better in the future
    else:
        print(f"Take a look at enzyme {enzyme_code}. There are fewer MRC ",
              "files than CSV files. That's weird!")
        return None


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

        # if the data extractor fails on that dataset
        if file_names is None:
            continue

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


def train_one_epoch(model,
                    data_loader,
                    optimizer,
                    criterion,
                    device,
                    epoch_num,
                    box_size=64):
    """
    Trains the model by one epoch

    Args:
        model (torch.nn.Module): The NN model to be trained.
        data_loader (torch.utils.data.DataLoader): DataLoader for the
            training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for
            upcoming weights.
        criterion (callable): The loss function. Note that often the loss is
            calculated inside the model during training. So this will often be
            set to None.
        device (torch.device): The device (CPU or CUDA) to run training on.
        epoch_num (int): The current epoch number (zero indexed).
        box_size (int): The pixel width/height to create bounding boxes from
            (x, y) coordinates.
    """

    # Set model to train mode
    model.train()

    # Initialize loss trackers
    # 'running_loss' adds the loss over mini-batches for periodic logging
    # 'total_loss' adds up the loss for the entire epoch
    running_loss = 0.0
    total_loss = 0.0

    # Print epoch header
    print(f"\n----- Epoch {epoch_num + 1} (training with {len(data_loader)}",
          " batches) -----")

    # Set up a few constants
    batches_before_printing = 3
    STACKED_MICROGRAPH_IMAGES_INDEX = 0
    COORDINATES_INDEX = 1

    # Loop over all batches of data
    for i, data_batch in enumerate(data_loader):
        # 'inputs' are the micrograph image tensors with shape (B, C, H, W)
        # B: number of images in my mini-batch
        # C: number of color channels (for our images, this is 1)
        # H: height, W: width (our images have resolution 4096 x 4096)
        # Move them to the target device
        inputs = data_batch[STACKED_MICROGRAPH_IMAGES_INDEX].to(device)

        # 'raw_coordinates_list' is a list of (N, 2) tensors where N is the
        # number of particles in that image.
        raw_coordinates_list = data_batch[COORDINATES_INDEX]

        # Dynamic size fit
        img_height = inputs.shape[2]
        img_width = inputs.shape[3]
        max_coord_x = img_width - 1
        max_coord_y = img_height - 1

        # Our CSVs only have (x, y) centers, so we need to build "boxes"
        # around them so they have some thickness
        targets = []
        valid_inputs = []
        for img_tensor, coords_tensor in zip(inputs, raw_coordinates_list):

            # Move the coordinate tensor to the device
            coords_tensor = coords_tensor.to(device)

            # We'll build a box of size (box_size, box_size) and place it
            # on each (x, y) coordinate
            boxes = torch.zeros((len(coords_tensor), 4),
                                dtype=torch.float32,
                                device=device)

            if len(coords_tensor) > 0:
                # FIXME: Replace these magic numbers
                # Note that "//" is integer (or floor) division
                boxes[:, 0] = coords_tensor[:, 0] - box_size // 2  # x_min
                boxes[:, 1] = coords_tensor[:, 1] - box_size // 2  # y_min
                boxes[:, 2] = coords_tensor[:, 0] + box_size // 2  # x_max
                boxes[:, 3] = coords_tensor[:, 1] + box_size // 2  # y_max
            else:
                pass

            # Clamp boxes to be within image bounds
            # (remember we set our image size to 4096)
            # Select all x coordinates (x_min and x_max)
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=max_coord_x)
            # Select all y coordinates (y_min and y_max)
            boxes[:, 1::3] = boxes[:, 1::3].clamp(min=0, max=max_coord_y)

            # Filter out boxes that have zero area
            # (may have resulted from clamping)
            valid_boxes_mask = (
                (boxes[:, 2] > boxes[:, 0]) &
                (boxes[:, 3] > boxes[:, 1])
            )
            boxes = boxes[valid_boxes_mask]

            # If all boxes are invalid, skip this image
            if len(boxes) == 0:
                continue

            # Label the boxes
            # The '0' class is reserved for background,
            # so we use '1' for the classification.
            labels = torch.ones((len(boxes),),
                                dtype=torch.int64,
                                device=device)

            # Add the data for this image to our list of targets
            targets.append({
                'boxes': boxes,
                'labels': labels
            })

            valid_inputs.append(img_tensor)

        # If there are no valid targets in this batch, skip it
        if len(valid_inputs) == 0:
            print(f"   Skipping batch {i} due to no valid targets after the ",
                  "boxes were created.")
            continue

        inputs = torch.stack(valid_inputs)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Now do the actual training
        try:
            # This is the standard way to train torchvision detection models
            loss_dict = model(inputs, targets)

            # The returned loss_dict may contain multiple losses.
            # So we add them up to get a single total loss for backpropagation.
            losses = sum(loss for loss in loss_dict.values())

            # Make sure the loss is a valid number
            if not torch.isfinite(losses):
                print(f"   Warning: Non-finite loss encountered: ,"
                      f"{losses.item()} at batch {i}. Skipping.")
                continue

        except Exception as e:
            print(f"       Batch {i + 1} failed with error: {e}")
            continue

        # Backward pass: this computes the gradient of the 'losses' tensor
        # with respect to all model parameters (weights and biases).
        losses.backward()

        # Update weights
        optimizer.step()

        # Update and log loss
        batch_loss = losses.item()
        running_loss += batch_loss
        total_loss += batch_loss

        # Print loss statistics every couple of batches
        if (i + 1) % batches_before_printing == 0:
            avg_running_loss = running_loss / batches_before_printing
            print(f"    [Batch {i + 1:5d} / {len(data_loader)}] running ,"
                  f"loss: {avg_running_loss:.4f}")
            running_loss = 0.0  # Reset the running loss tracker

    # Return average loss for the epoch
    avg_epoch_loss = total_loss / len(data_loader)
    return avg_epoch_loss


# This "main" block is mostly just to test specific functions
if __name__ == "__main__":
    thing = find_all_data()
    print(thing[1][4])
