import os
import numpy as np
import nrrd
import tkinter as tk
from tkinter import filedialog

import usefull_function as uf
import presentation
# this module used to compare the automatic segmentation
# with manual segmentation made by the software 3d slicer

def choose_nrrd_file():
    """
    Opens a file dialog for the user to choose a NRRD file.

    Returns:
        str: The selected NRRD file path or an empty string if no file is selected.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
        title="Select NRRD File",
        filetypes=[("NRRD files", "*.nrrd"), ("All files", "*.*")]
    )

    return file_path


# Function for saving automated ROI images and slices to simplify the manual segmentation process
def update_image_to_dicom(slices, roi_images, new_path=""):
    """
    Save updated images and their size (rows & columns) as DICOM files.

    :param slices: List of DICOM images.
    :param roi_images: List of region of interest (ROI) images corresponding to the DICOM images.
    :param new_path: Path to save the updated DICOM files.
    :return: None
    """
    # If new_path is not provided, prompt the user to enter a path
    if new_path == "":
        new_path = input("Enter a path for the updated DICOM files: ")

    # Create a new folder in the specified path
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    os.chdir(new_path)

    # Convert ROI images to 16-bit unsigned integer numpy arrays
    np_roi_images = [np.array(i).astype(np.uint16) for i in roi_images]

    # Update DICOM images with new size and pixel data
    images = [slices[i] for i in range(len(np_roi_images))]
    for i in range(len(images)):
        images[i].Rows, images[i].Columns = roi_images[i].shape
        images[i].PixelData = np_roi_images[i].tobytes()
        images[i].save_as(str(i) + ".dcm")


# Function to read and return data from a "nrrd" file with optional dimension swapping
# The manual segmentations are a "nrrd" file
def exporting_nrrd(path="", fix_dim=False):
    """
    Read and return data from a "nrrd" file with an option to fix dimensions.

    :param path: Path to the "nrrd" file.
    :param fix_dim: Boolean flag to swap dimensions if True.
    :return: numpy array.
    """
    if path == "":
        path = choose_nrrd_file()

    # Read nrrd data and header
    data, header = nrrd.read(path)

    # Swap dimensions if fix_dim is True
    # the axial and sagittal were usually swapped in the manual segmentation
    if fix_dim:
        data = data.swapaxes(0, 2)

    return data



# Function to perform validation tests on segmentation results
def validation_test(ground_truth, auto_segmentation):
    """
    Compare two segmentations, count TP, FP, and FN, calculate Dice, and sensitivity.

    :param ground_truth: Binary array representing the ground truth segmentation.
    :param auto_segmentation: Binary array representing the automated segmentation.
    :return: Tuple containing Dice coefficient, sensitivity, and a tuple with TP, FP, and FN counts.
    """
    # Binary arrays for true positive (TP), false positive (FP), and false negative (FN)
    X = (auto_segmentation > 0)
    Y = (ground_truth > 0)
    X_and_Y = np.logical_and(X, Y)

    # Calculate Dice coefficient, FP, FN, TP, and sensitivity
    D = round(2 * np.sum(X_and_Y) / (np.sum(X) + np.sum(Y)), 3) * 100
    FP = np.sum(np.logical_and(X, np.logical_not(Y)))
    FN = np.sum(np.logical_and(Y, np.logical_not(X)))
    TP = np.sum(X_and_Y)
    sensitivity = round(TP / (TP + FN), 3) * 100

    # to calculate accuracy
    # X_or_Y = np.logical_or(X, Y)
    # X_nor_Y = np.logical_not(X_or_Y)
    # TN = np.sum(X_nor_Y)
    # num_of_pixels = X.size

    # Return Dice, sensitivity, and counts of TP, FP, FN
    return D, sensitivity, (TP, FP, FN)


def test(auto_segmentation, ground_truth, start=0, end=-1, inferior=0, vol_factor=1):
    """
    Validate an automated segmentation with a manual segmentation using Dice and sensitivity.
    Also, evaluate volume differences. Can validate a specific region by specifying start and end coronal indexes,
    and inferior as the axial index to begin.

    :param auto_segmentation: Automated segmentation images.
    :param start: Starting coronal index for validation.
    :param end: Ending coronal index for validation.
    :param inferior: Axial index where validation begins.
    :param vol_factor: Volume scaling factor.
    :return: None
    """

    D, sensitivity, (TP, FP, FN) = validation_test(ground_truth[:, start:end, inferior:], auto_segmentation[:, start:end, inferior:])
    print("Dice:", D, " | Sensitivity:", sensitivity, " | FP volume:", FP * vol_factor, " | FN volume:", FN * vol_factor)


# Saving the images that compare the manual and automatic segmentations.
# The identical pixels in both (TF) are marked in green.
# The pixels detected only in the manual segmentation (FN) are marked in blue,
# And the pixels detected only in the automatic segmentation (FP) are marked in red.
def compare_segmentation(slices, roi_images, segmentation, ground_truth, start=0, end=-1):
    """
    Compare manual and automatic segmentations, saving the result images.

    :param slices: CT slices containing the airway.
    :param roi_images: ROI slices and images.
    :param segmentation: Automated segmented airway.
    :param ground_truth: Manual segmented airway.
    :param start: Coronal index to begin the comparison images.
    :param end: Coronal index to end the comparison images.
    :return: List of presentation images, pixels data, axial data, and coronal data.
    """

    # HU parameters for rescaling the image
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    X = np.copy(segmentation)
    Y = np.copy(ground_truth)


    tp = np.logical_and(X, Y).astype(int) * 255
    fp = np.logical_and(X, np.logical_not(Y)).astype(int) * 255
    fn = np.logical_and(Y, np.logical_not(X)).astype(int) * 255

    # Data for understanding
    pixels_data = [np.count_nonzero(x) for x in (tp[:, start:end, :], fp[:, start:end, :], fn[:, start:end, :])]
    axial_data = []
    coronal_data = []
    for x in (fp, fn):
        x = x[:, start:end + 1, :]
        slices_pixel_data_temp = [np.count_nonzero(x[:, i, :]) for i in range(x.shape[1])]
        axial_data.append(slices_pixel_data_temp)
        slices_pixel_data_temp = [np.count_nonzero(x[:, :, i]) for i in range(x.shape[2])]
        coronal_data.append(slices_pixel_data_temp)

    # Gray to color transform
    tp = uf.np_3D_to_color(tp)
    fp = uf.np_3D_to_color(fp)
    fn = uf.np_3D_to_color(fn)
    color_roi = uf.gray_to_color(roi_images, intercept=intercept, slope=slope)

    color_segmentation = np.copy(color_roi)
    color_segmentation[np.where((tp == [255, 255, 255]).all(axis=3))] = [0, 255, 0]  # True positive in green
    color_segmentation[np.where((fp == [255, 255, 255]).all(axis=3))] = [255, 0, 0]  # False positive in red
    color_segmentation[np.where((fn == [255, 255, 255]).all(axis=3))] = [0, 0, 255]  # False negative in blue
    presentation_images = []
    for i in range(np.shape(color_roi)[0]):
        img_a = color_roi[i]
        img_b = color_segmentation[i]
        image = presentation.connect_images((img_a, img_b))
        presentation_images.append(image)
    return presentation_images, pixels_data, axial_data, coronal_data
