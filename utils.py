import os
import zipfile
import random
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import numpy as np
import cv2

preprocessed_folder = "uploads/"
intermediate_folder = "heatmaps/"
segmentation_folder = "segmentations/"
tables_folder = "tables/"
cell_descriptors_path = "cell_descriptors/cell_descriptors.csv"
zip_file_path = "outputs.zip"


def select_sample_images():
    # first check if a zip file has been uploaded and extract images from it
    for file_name in os.listdir(preprocessed_folder):
        if file_name.endswith(".zip"):
            zip_file_path = os.path.join(preprocessed_folder, file_name)
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                # Extract all contents of the zip file to the preprocessed_folder
                zip_ref.extractall(path=preprocessed_folder)
            # Remove the original zip file
            os.remove(zip_file_path)
            # print("Contents of the zip file extracted to the folder.")
            break

    # Get a list of all subfolders in the main folder
    subfolders = [
        f
        for f in os.listdir(preprocessed_folder)
        if os.path.isdir(os.path.join(preprocessed_folder, f))
    ]
    # Iterate through each subfolder and move its files to the main folder
    for subfolder in subfolders:
        if "MACOSX" not in subfolder:
            subfolder_path = os.path.join(preprocessed_folder, subfolder)
            for file_name in os.listdir(subfolder_path):
                source_path = os.path.join(subfolder_path, file_name)
                destination_path = os.path.join(preprocessed_folder, file_name)
                shutil.move(source_path, destination_path)
            # print(f"Moved file '{file_name}' from '{subfolder}' to '{main_folder}'")
    # Delete empty subfolders
    for subfolder in subfolders:
        if "MACOSX" not in subfolder:
            subfolder_path = os.path.join(preprocessed_folder, subfolder)
            try:
                os.rmdir(subfolder_path)
                # print(f"Deleted empty folder '{subfolder}'")
            except OSError as e:
                print(f"Error deleting folder '{subfolder}': {e}")

    # next check the count of images in the folder
    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
    ]  # Add more extensions if needed
    image_count = 0
    for file_name in os.listdir(preprocessed_folder):
        if any(file_name.lower().endswith(ext) for ext in image_extensions):
            image_count += 1

    # if count > 5, return 5 random indices
    # else, return all 5 indices
    if image_count > 5:
        indices = random.sample(range(image_count), 5)
        indices.sort()
        return indices
    else:
        return list(range(image_count))


def create_cell_descriptors_table(table_path, nucleus_area, cytoplasm_area, ratio):
    # Sample data for the table
    data = {
        "Metric": ["Nucleus Area", "Cytoplasm Area", "N:C Ratio"],
        "Value": [
            str(round(nucleus_area, 5)),
            str(round(cytoplasm_area, 5)),
            str(round(ratio, 5)),
        ],
    }

    # Define cell colors
    cell_colors = [
        ["lightgrey", "lightblue"],
        ["lightgrey", "lightgreen"],
        ["lightgrey", "lightyellow"],
    ]

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Plot table
    fig = plt.figure(figsize=(2, 2))
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        cellColours=cell_colors,
    )

    # Set cell heights
    table.auto_set_font_size(False)
    table.set_fontsize(6)  # Adjust font size if needed
    table.scale(1, 2)  # Increase cell heights

    # Hide axes
    plt.axis("off")

    fig.tight_layout()
    # Save as image
    fig.savefig(table_path)  # pad_inches=(0.1, 0.1, 0.1, 0.1)  bbox_inches="tight"
    plt.close()
    # plt.show()


def delete_folders(folder_names):
    for folder_name in folder_names:
        try:
            shutil.rmtree(folder_name)
            # print(f"Folder deleted: {folder_name}")
        except FileNotFoundError:
            print(f"Folder does not exist: {folder_name}")
        except Exception as e:
            print(f"Error deleting folder {folder_name}: {e}")


def create_folders(folder_names):
    for folder_name in folder_names:
        try:
            os.makedirs(folder_name)
            # print(f"Folder created: {folder_name}")
        except FileExistsError:
            print(f"Folder already exists: {folder_name}")
        except Exception as e:
            print(f"Error creating folder {folder_name}: {e}")


def calculate_cell_descriptors(
    original_shape, resized_shape, pixel_conversion, segmentation_mask
):
    area_of_pixel = (
        original_shape[0]
        * original_shape[1]
        * (pixel_conversion**2)
        / (resized_shape[0] * resized_shape[1])
    )

    binary_nucleus = np.zeros(resized_shape, dtype=np.uint8)
    binary_cytoplasm = np.zeros(resized_shape, dtype=np.uint8)
    binary_nucleus[(segmentation_mask == [255, 0, 0]).all(axis=2)] = 1
    binary_cytoplasm[(segmentation_mask == [128, 0, 0]).all(axis=2)] = 1

    # Find contours in the binary masks
    nucleus_contours, _ = cv2.findContours(
        binary_nucleus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # nucleus
    cytoplasm_contours, _ = cv2.findContours(
        binary_cytoplasm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # cytoplasm

    # Calculate area for nucleus and cytoplasm
    nucleus_area = sum(cv2.contourArea(contour) for contour in nucleus_contours)
    cytoplasm_area = sum(cv2.contourArea(contour) for contour in cytoplasm_contours)
    if cytoplasm_area == 0:
        ratio = np.NaN
    else:
        ratio = nucleus_area / cytoplasm_area

    return nucleus_area * area_of_pixel, cytoplasm_area * area_of_pixel, ratio


def create_zip_file():
    folders = [intermediate_folder, segmentation_folder]
    csv_file = cell_descriptors_path
    with zipfile.ZipFile(zip_file_path, "w") as zipf:
        # Add folders to the zip file
        for folder in folders:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.join(folder, ".."))
                    zipf.write(file_path, arcname=arcname)

        # Add the CSV file to the zip file
        zipf.write(csv_file, arcname=os.path.basename(csv_file))
