import os
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import exposure
from cellpose import models
import numpy as np
import pandas as pd
from spotiflow.model import Spotiflow


def process_images(directory, image_save_dir, save_dir):
    """
    Processes a set of images in a given directory, performs cell segmentation and spot detection,
    and saves the processed results to specified directories.

    Parameters:
    -----------
    directory : str
        The path to the directory containing the image files. The images should be in `.tif` format 
        and organized such that corresponding C1 and C2 channels are named similarly (e.g., `C1_image1.tif` 
        and `C2_image1.tif`).

    image_save_dir : str
        The path to the directory where processed images (e.g., mask images and spot visualizations) 
        will be saved. Each image will be saved in a subfolder named after the corresponding C1 image 
        (e.g., `C1_image1/` for `C1_image1.tif`).

    save_dir : str
        The path to the directory where CSV files containing the analysis results will be saved.
        Two CSV files will be created for each image:
        - `<image_name>_spots.csv` containing spot counts for each mask
        - `<image_name>_intensities.csv` containing average intensities for each 
    
    Example Usage:
    ---------------
    ```python
    process_images("/path/to/images", "/path/to/save/images", "/path/to/save/csv")
    ```
    This will process all the images in `/path/to/images`, save the results in `/path/to/save/images`, 
    and save the CSV files in `/path/to/save/csv`.
    """
    
    base_model = models.Cellpose(gpu=True, model_type="cyto3")
    model_spot = Spotiflow.from_pretrained("general")

    image_folder = os.listdir(directory)

    count = 0
    flow_threshold = 0.4
    cellprob_threshold = 0

    for i in range(len(image_folder)):
        if image_folder[i].startswith('C1'):
            os.mkdir(image_save_dir + image_folder[i])

            df_c1 = pd.DataFrame()
            df_c2 = pd.DataFrame()

            count += 1
            image = imread(os.path.join(directory, image_folder[i]))
            image_c2 = imread(os.path.join(directory, image_folder[i].replace('C1', 'C2')))

            for t in range(len(image)):
                image_t = image[t, :, :]
                image_c2_t = image_c2[t, :, :]

                df_c1[str(t)] = [0] * 100
                df_c2[str(t)] = [0] * 100

                masks, _, _ = base_model.eval(
                    image_t,
                    diameter=200,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold
                )

                # Spot detection
                spots, details = model_spot.predict(image_t, verbose=0)
                spots = spots.astype(int)
                unique_masks = np.unique(masks)
                unique_masks = unique_masks[unique_masks > 0]

                mask_coordinates = {mask: 0 for mask in unique_masks}

                for coord in spots:
                    x, y = coord
                    mask_value = masks[x, y]
                    if mask_value > 0:  # Exclude background
                        mask_coordinates[mask_value] += 1

                mask_coordinates_list = [count for _, count in mask_coordinates.items()]
                df_c1.loc[range(len(mask_coordinates_list) + 1), str(t)] = mask_coordinates_list + [-1]

                intensities = []
                for value in unique_masks:
                    avg = np.sum(image_c2_t[masks == value]) / len(image_c2_t[masks == value])
                    intensities.append("{:.2f}".format(avg))

                df_c2.loc[range(len(mask_coordinates_list) + 1), str(t)] = intensities + [-1]

                # Save mask image
                plt.imsave(os.path.join(image_save_dir, image_folder[i], f"mask_{t}.png"), masks)

                # Save spot visualization
                fig, ax = plt.subplots()
                p2, p98 = np.percentile(image_t, (2, 98))
                img_rescale = exposure.rescale_intensity(image_t, in_range=(p2, p98))
                ax.imshow(img_rescale)
                for y, x in spots:
                    circle = plt.Circle((x, y), 2, color="red")
                    ax.add_patch(circle)
                fig.savefig(os.path.join(image_save_dir, image_folder[i], f"spot_{t}.png"))
                plt.close()

            df_c1.to_csv(os.path.join(save_dir, image_folder[i].replace('.tif', '_spots.csv')))
            df_c2.to_csv(os.path.join(save_dir, image_folder[i].replace('.tif', '_intensities.csv')))

            print(f"Processed: {count} - {image_folder[i]}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process images for spot detection and intensity analysis.")
    
    # Define arguments for directories
    parser.add_argument('--directory', type=str, required=True, help="Path to the directory containing images")
    parser.add_argument('--image_save_dir', type=str, required=True, help="Path to the directory to save processed images")
    parser.add_argument('--save_dir', type=str, required=True, help="Path to the directory to save CSV files")

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Get the directory paths from arguments
    directory = args.directory
    image_save_dir = args.image_save_dir
    save_dir = args.save_dir

    # Run the image processing
    process_images(directory, image_save_dir, save_dir)
