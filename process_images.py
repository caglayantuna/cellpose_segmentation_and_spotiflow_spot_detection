import os
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import exposure
from cellpose import models
import numpy as np
import pandas as pd
from spotiflow.model import Spotiflow


def process_images(directory, image_save_dir, save_dir):
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


if __name__ == "__main__":
    directory = "/path/to/images/"
    image_save_dir = "/path/to/image_save_dir/"
    save_dir = "/path/to/save_dir/"

    process_images(directory, image_save_dir, save_dir)