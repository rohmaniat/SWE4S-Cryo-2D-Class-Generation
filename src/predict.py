import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import mrcfile
import pandas as pd
import numpy as np
import argparse
import time
import warnings
import matplotlib.pyplot as plt
import configparser


def load_model(model_path, device):
    # Helper function to create the model and load the weights
    # Remake the model architecture
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,  # We're loading our own weights
        weights_backbone=None,
        image_mean=[0.449],
        image_std=[0.226]
    )

    # Change the input layer for grayscale
    model.backbone.body.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    # Change the output head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    NUM_CLASSES = 2  # 0 = background, 1 = particle
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      NUM_CLASSES)

    # Load in the saved weights
    model.load_state_dict(torch.load(model_path,
                                     map_location=device,
                                     weights_only=True))

    # Set to evaluation mode and move to the device
    model.to(device)
    model.eval()
    return model


def load_and_transform_mrc(mrc_path, device):
    """Loads and preprocesses an MRC file."""

    # Define the exact same transforms from training
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((800, 800), antialias=True),  # Resize images
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # This converts the Numpy array to a Tensor
    ])

    # Open the MRC file (with a warnings catch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            image_data = mrc.data
            if image_data.ndim > 2:
                print(f"Warning: Input is a stack ({image_data.shape}).",
                      "Using first image only.")
                image_data = image_data[0]

    # Store original dimensions for scaling coordinates later
    original_height, original_width = image_data.shape

    # Normalize image to 0-255 range and cast to uint8
    # This matches what ToPILImage() expects
    img_min = image_data.min()
    img_max = image_data.max()
    image_data = (image_data - img_min) / (img_max - img_min) * 255.0
    image = image_data.astype(np.uint8)

    # Apply transforms and add to device
    # The model expects a list of tensors, not a batched tensor
    image_tensor = data_transform(image).to(device)

    return [image_tensor], original_height, original_width


def create_visualization(mrc_path,
                         predictions_df,
                         ground_truth_df,
                         output_image_path):
    # FIXME: add threshold argument
    """
    Saves a PNG of the micrograph with ground truth and predictions plotted
    """
    # Load micrograph
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            image_data = mrc.data
            if image_data.ndim > 2:
                image_data = image_data[0]

    # Normalize for plotting
    img_min = image_data.min()
    img_max = image_data.max()
    image_norm = (image_data - img_min) / (img_max - img_min)

    # Convert to 3 channel RGB
    image_rgb = np.stack([image_norm, image_norm, image_norm], axis=-1)

    # Create plot
    fig, ax = plt.subplots(1, figsize=(15, 15))
    ax.imshow(image_rgb, cmap='gray')

    # Plot ground truth
    if ground_truth_df is not None and not ground_truth_df.empty:
        ax.scatter(
            ground_truth_df['coord_x'],
            ground_truth_df['coord_y'],
            marker='x', s=100,
            color="blue",
            label="Ground Truth",
            linewidths=2,
            alpha=0.7
        )

    # Plot predictions
    if not predictions_df.empty:
        ax.scatter(
            predictions_df['coord_x'],
            predictions_df['coord_y'],
            marker='+', s=100,
            color="red",
            label=f"Prediction (score > {args.threshold})",
            linewidths=2,
            alpha=0.7
        )

    # Finalize and save
    ax.legend()
    ax.set_axis_off()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    # Build configuration parser
    config = configparser.ConfigParser()

    # Build argument parser
    parser = argparse.ArgumentParser(
        description="Predict particles from micrographs using "
        + "a provided model."
    )
    parser.add_argument("--config", type=str,
                        help="Path to a .ini configuration file.")
    parser.add_argument("--mrc_file", type=str,
                        help="Path to the input .mrc file.")
    parser.add_argument("--output_csv", type=str,
                        help="Path to save the output .csv file.")
    parser.add_argument("--model_path", type=str,
                        default="src/models/my_model.pt",
                        help="Path to trained .pt model file.")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Confidence threshold for keeping predictions "
                        + "(0.0 to 1.0).")
    parser.add_argument("--output_image", type=str, default=None,
                        help="[Optional] Path to save a PNG "
                        + "visualizing the predictions.")
    parser.add_argument("--ground_truth_csv", type=str, default=None,
                        help="[Optional] Path to the ground truth "
                        + "CSV for visualization.")
    args = parser.parse_args()

    # Manage passed configuration file
    if args.config:
        config.read(args.config)

        def get_config_val(section, key, type_func=str):
            cli_val = getattr(args, key)  # Get value from the command line
            if cli_val is not None:
                return cli_val  # Return CLI value if it exists

            try:
                config_val = config.get(section, key)
                if config_val == "":
                    return None
                return type_func(config_val)
            except (configparser.NoOptionError, configparser.NoSectionError):
                return None

        # Apply this logic for all arguments
        args.mrc_file = get_config_val('paths', 'mrc_file', str)
        args.output_csv = get_config_val('paths', 'output_csv', str)
        args.model_path = get_config_val('paths', 'model_path', str)
        args.output_image = get_config_val('paths', 'output_image', str)
        args.ground_truth_csv = get_config_val('paths', 'ground_truth_csv',
                                               str)
        args.threshold = get_config_val('settings', 'threshold', float)

    # Validate that all requred arguments are present
    # (either from the CLI or config.ini)
    required_args = ['mrc_file', 'output_csv', 'model_path', 'threshold']
    missing_args = [
        arg for arg in required_args if getattr(args, arg) is None
    ]
    if missing_args:
        print(f"\nError: The following required arguments are missing:")
        for arg in missing_args:
            print(f"  --{arg}")
            print(f"\nPlease provide them via the command line or ",
                  "a --config file.\n")
            parser.print_help()
            exit(1)  # Exit code 1 for missing required args

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}.")

    # Load in a model
    print(f"Loading model from {args.model_path}")
    try:
        model = load_model(args.model_path, device=device)
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Could not load model: {e}")
        exit(2)  # Exit code 2 for not being able to upload the model

    # Load and transform data
    print(f"Loading and processing micrograph: {args.mrc_file}")
    image_list, orig_h, orig_w = load_and_transform_mrc(args.mrc_file, device)

    # Run inference (prediction)
    print("Running prediction...")
    start_time = time.time()
    with torch.no_grad():
        predictions = model(image_list)
    end_time = time.time()
    print(f"Prediction finished in {end_time - start_time:.2f} seconds.")

    # 'predictions' is a list of dictionaries (we only sent one image)
    output = predictions[0]

    # Filter and process results
    scores = output['scores']
    boxes = output['boxes']
    labels = output['labels']

    # Keep predictions that pass the confidence threshold and are labeled 1
    keep_mask = (scores > args.threshold) & (labels == 1)

    final_boxes = boxes[keep_mask].cpu().numpy()
    final_scores = scores[keep_mask].cpu().numpy()

    # Initialize df as an empty DataFrame first
    df = pd.DataFrame(columns=['coord_x', 'coord_y', 'confidence'])

    if final_boxes.shape[0] > 0:
        # Scale coordinates from resized (800x800) to original
        scale_x = orig_w / 800.0
        scale_y = orig_h / 800.0

        # Get center of boxes [x1, y1, x2, y2]
        # FIXME: Remove these magic numbers
        center_x = (final_boxes[:, 0] + final_boxes[:, 2]) / 2
        center_y = (final_boxes[:, 1] + final_boxes[:, 3]) / 2

        # Scale centers
        scaled_center_x = center_x * scale_x
        scaled_center_y = center_y * scale_y

        # Save as CSV
        df_data = {
            'coord_x': scaled_center_x.astype(int),
            'coord_y': scaled_center_y.astype(int),
            'confidence': final_scores
        }
        df = pd.DataFrame(df_data)
        df.to_csv(args.output_csv, index=False)
        print(f"Success! Saved {len(df)} particle coordinates",
              f"to {args.output_csv}")

    else:
        print(f"No particles found with confidence above {args.threshold}")

    # Save data
    df.to_csv(args.output_csv, index=False)

    # Visualization (if requested)
    if args.output_image:
        print(f"\nCreating visualization at {args.output_image}...")

        # Load ground truth if provided
        gt_df = None
        if args.ground_truth_csv:
            try:
                gt_df = pd.read_csv(args.ground_truth_csv)

                required_cols = {"X-Coordinate", "Y-Coordinate"}
                if required_cols.issubset(gt_df.columns):
                    gt_df = gt_df.rename(columns={
                        "X-Coordinate": 'coord_x',
                        "Y-Coordinate": "coord_y"
                    })
                    print(f"Loaded {len(gt_df)} ground truth",
                            f"coordinates from {args.ground_truth_csv}")
                elif ('coord_x' not in gt_df.columns or
                      "coord_y" not in gt_df.columns):
                    print(f"Warning: Ground truth CSV ",
                          f"({args.ground_truth_csv})")
                    print(f"         does not contain 'X-Coordinate'/",
                          "'Y-Coordinate'.")
                    print(f"         Ground truth will not be plotted")
                    gt_df = None
                else:
                    print(f"Loaded {len(gt_df)} ground truth coordinates",
                          f"from {args.ground_truth_csv}.")

            except Exception as e:
                print(f"Warning: Could not load ground truth CSV: {e}")
                print("Will proceed with predictions only.")

        # Call the visualization function
        create_visualization(
            mrc_path=args.mrc_file,
            predictions_df=df,
            ground_truth_df=gt_df,
            output_image_path=args.output_image
        )
        print("Visualization saved!")
