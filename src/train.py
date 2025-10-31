import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from class_def import CryoEMDataset
from class_def import collate
import utils
from utils import train_one_epoch
import torch.optim as optim
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

if __name__ == "__main__":

    # If there is a CUDA-compatible GPU is available, use that.
    # If not, just use the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Let's get all the filepaths
    mrc_paths, csv_paths = utils.find_all_data()
    print(f"Found {len(mrc_paths)} total samples.")

    # Build the transformation pipeline
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((800, 800), antialias=True),  # Resize images
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # This converts the Numpy array to a Tensor
    ])

    # Create an instance of the CryoEMDataset class
    cryo_dataset = CryoEMDataset(
        mrc_paths=mrc_paths,
        csv_paths=csv_paths,
        transform=data_transform
    )

    # Build the dataloader
    data_loader = DataLoader(cryo_dataset,
                             batch_size=2,
                             shuffle=True,
                             collate_fn=collate,
                             num_workers=0
                             )

    # Load in a pre-trained model (transfer learning for the win!)
    image_mean = [0.449]
    image_std = [0.226]
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        image_mean=image_mean,
        image_std=image_std
    )

    # This pre-existing model was trained on colored data.
    # Our images are grayscaled, so we need to change the input to only
    # one channel.
    model.backbone.body.conv1 = torch.nn.Conv2d(
        1,  # This is in_channels (set to 1 because we have grayscale images)
        64,  # out_channels
        kernel_size=(7, 7),  # The "window" size that slides over the image.
        stride=(2, 2),  # The window moves two pixels at a time.
        padding=(3, 3),  # Adds a 3-pixel border of zeros around the image.
        bias=False
    )

    # Get the number of input features for the classifier.
    # We can't use all 4096x4096 pixels and pass them into the NN.
    # There already exists an internal step that converts the image into a
    # feature vector.
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Define the number of classes
    # Remember 0 = "background" and 1 = "particle"
    num_classes = 2

    # Replace the pre-trained head with a new, untrained one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      num_classes)

    # Move the model to the device
    model.to(device)
    print("Model setup complete. Using Faster R-CNN with ResNet50 backbone.")

    # Get all the parameters from the model that require gradients
    params = [p for p in model.parameters() if p.requires_grad]

    # Create the optimizer. Adam is a good, modern default
    # The learning rate (lr) dictates how quickly the model will change.
    # Setting a low lr is good for fine-tuning a model.
    optimizer = optim.Adam(params, lr=0.0001)

    # The loss is calculated inside the model when we pass it targets,
    # so we don't need a separate criterion function.
    criterion = None

    # THE MAIN TRAINING LOOP
    num_epochs = 2
    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch_num=epoch
        )

        print(f"  -- Epoch {epoch + 1} of {num_epochs} complete. ",
              "Average epoch loss: {avg_loss}. --")

    print(f"\n---- Training finished! ----\n")

    # Now save the model
    model_name = "my_model"
    model_save_path = "src/models/" + model_name
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
