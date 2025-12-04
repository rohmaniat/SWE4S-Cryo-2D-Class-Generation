import torch
from torch.utils.data import DataLoader, random_split
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

    img_size = 1024

    # Build the transformation pipeline
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # This converts the Numpy array to a Tensor
    ])

    # Create an instance of the CryoEMDataset class
    cryo_dataset = CryoEMDataset(
        mrc_paths=mrc_paths,
        csv_paths=csv_paths,
        transform=data_transform
    )

    train_size = int(0.9 * len(cryo_dataset))
    val_size = len(cryo_dataset) - train_size
    train_dataset, val_dataset = random_split(
        cryo_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Build the dataloader
    data_loader = DataLoader(cryo_dataset,
                             batch_size=4,  # can decrease to 2
                             shuffle=True,
                             collate_fn=collate,
                             num_workers=0
                             )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
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
    model.backbone.body.conv1 = torch.nn.Conv2d(
        1,  # This is in_channels (set to 1 because we have grayscale images)
        64,  # out_channels
        kernel_size=(7, 7),  # The "window" size that slides over the image.
        stride=(2, 2),  # The window moves two pixels at a time.
        padding=(3, 3),  # Adds a 3-pixel border of zeros around the image.
        bias=False
    )

    # Get the number of input features for the classifier.
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
    optimizer = optim.Adam(params, lr=0.0001, weight_decay=0.0005)

    criterion = None

    # scheduler optimizes learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # THE MAIN TRAINING LOOP
    num_epochs = 10
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

        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        # training logs after each epoch to calculate loss
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss,
            }

        torch.save(checkpoint, "src/checkpoint_epoch.pth")

        print(f"  Running validation...")

        with torch.no_grad():
            for val_batch in val_loader:
                if val_batch is None:
                    continue

                val_images = val_batch[0].to(device)
                val_coords = val_batch[1]

                # Build targets (same logic as training)
                val_targets = []
                valid_val_images = []

                for img_tensor, coords_tensor in zip(val_images, val_coords):
                    coords_tensor = coords_tensor.to(device)

                    if len(coords_tensor) == 0:
                        continue

                    boxes = torch.zeros((len(coords_tensor), 4),
                                        dtype=torch.float32,
                                        device=device)

                    box_size = 64
                    boxes[:, 0] = coords_tensor[:, 0] - box_size // 2
                    boxes[:, 1] = coords_tensor[:, 1] - box_size // 2
                    boxes[:, 2] = coords_tensor[:, 0] + box_size // 2
                    boxes[:, 3] = coords_tensor[:, 1] + box_size // 2

                    # FIX: Correct clamping
                    max_coord_x = img_size - 1
                    max_coord_y = img_size - 1
                    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0,
                                                          max=max_coord_x)
                    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0,
                                                          max=max_coord_y)

                    valid_mask = (
                        boxes[:, 2] > boxes[:, 0]) & (
                            boxes[:, 3] > boxes[:, 1])
                    boxes = boxes[valid_mask]

                    if len(boxes) == 0:
                        continue

                    labels = torch.ones((len(boxes),),
                                        dtype=torch.int64,
                                        device=device)
                    val_targets.append({'boxes': boxes, 'labels': labels})
                    valid_val_images.append(img_tensor)

                if len(valid_val_images) == 0:
                    continue

                val_images_stacked = torch.stack(valid_val_images)

                try:
                    loss_dict = model(val_images_stacked, val_targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
                    num_val_batches += 1
                except Exception as e:
                    continue

        if num_val_batches > 0:
            avg_val_loss = val_loss / num_val_batches
            print(f"  Validation loss: {avg_val_loss:.4f}")

            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = "src/models/best_model.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"  ** New best model saved! **")

        model.train()  # Switch back to training mode

    print(f"\n---- Training finished! ----\n")
    print(f"Best validation loss: {best_val_loss:.4f}\n")

    # Now save the model
    model_name = "new_model"
    model_save_path = "src/models/" + model_name
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
