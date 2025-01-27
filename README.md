# Cell Segmentation and Spot Detection

This repository contains code and notebooks for cell segmentation and spot detection. It uses pre-trained models from **Cellpose** for cell segmentation and **Spotiflow** for spot detection.

### Files in this repository:
1. **cellpose_segmentation.ipynb**  
   Demonstrates how to segment cells using pre-trained models from Cellpose.

2. **spotiflow_spot_detection.ipynb**  
   Shows how to detect spots within segmented cells using pre-trained models from Spotiflow.

3. **process_images.py**  
   A Python script to apply the segmentation and spot detection methods to all images in a directory. The script will:
   - Segment cells using Cellpose
   - Detect spots within segmented cells using Spotiflow
   - Count the number of spots per cell and calculate the mean intensity
   - Store the results in a CSV file.

### Requirements:
- Python 3.x
- Required libraries (listed in `requirements.txt`):
  - Cellpose
  - Spotiflow
  - Pandas
  - NumPy
  - Matplotlib

### How to Use:
1. Clone this repository to your local machine.
2. Install the dependencies using:
pip install -r requirements.txt
3. Run the notebooks or use the Python script (`process_images.py`) to process your images:
python process_images.py <path_to_images_directory> <output_csv_file>

### Output:
- The script will generate a CSV file containing:
- Cell ID
- Spot count for each cell
- Mean intensity of spots within each cell






