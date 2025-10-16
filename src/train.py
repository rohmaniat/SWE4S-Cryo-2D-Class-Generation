from torch.utils.data import DataLoader
from torchvision import transforms
from class_def import CryoEMDataset
from class_def import collate
import utils

if __name__ == "__main__":

    # First, let's get all the filepaths
    mrc_paths, csv_paths = utils.find_all_data()
    print(f"Found {len(mrc_paths)} total samples.")

    # Build the transformation pipeline
    data_transform = transforms.Compose([
        transforms.ToTensor(),  # This converts the Numpy array to a Tensor
        transforms.Resize((4096, 4096), antialias=True)  # Resize the Tensor
    ])

    # Create an instance of the CryoEMDataset class
    cryo_dataset = CryoEMDataset(
        mrc_paths=mrc_paths,
        csv_paths=csv_paths,
        transform=data_transform
    )

    # Build the dataloader
    data_loader = DataLoader(cryo_dataset, batch_size=4, shuffle=True, collate_fn=collate)

    # Check a sample batch  (everything below this line was from Gemini)
    first_batch_of_micrographs, first_batch_of_coords = next(iter(data_loader))

    print("\n--- Checking a sample batch ---")
    print(f"Micrographs batch shape: {first_batch_of_micrographs.shape}")
    print(f"Micrographs batch data type: {first_batch_of_micrographs.dtype}")

    print(f"Coordinates batch is a list containing {len(first_batch_of_coords)} tensors:")
    for i, coords_tensor in enumerate(first_batch_of_coords):
        print(f"  - Shape of coordinates for image {i}: {coords_tensor.shape}")
