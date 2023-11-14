import cv2 as cv
import numpy as np
import cc3d
import pandas as pd


CASE_MAP = {"2": "2", "3": "C5449", "4": "C4672", "5": "C2579", "6": "C1701", "7": "C3299", "8": "C5508", "9": "C7483",
            "10": "C6244",
            "11": "C8468", "12": "C9515_CL12279", "13": "CL11461_CL16234", "14": "CL2648", "15": "CL7196",
            "16": "CL6153", "17": "CL5525", "18": "CL4179",
            "19": "CL3187", "20": "CL8207", "21": "CL9224", "22": "CL10234", "23": "CL11165", "24": "CL12279",
            "25": "CL17202", "26": "CL14142", "27": "CL15153", "28": "CL13362"}


# help general function

def cc_stat(img, connectivity=8):
    """
    Give data about all the objects in the image using connected component labeling and analysis.
    :param img: 2D numpy.ndarray or 2D matrix - Input image
    :param connectivity: int - Connectivity value (default: 8)
    :return: tuple - Data about the objects in the image:
                     - Number of objects
                     - 2D numpy.ndarray with the labels of objects
                     - Data about each object (left, top, height, width, area)
                     - Centroid pixel (tuple of float)
    """
    return cv.connectedComponentsWithStats(np.uint8(img), connectivity=8)


def erase_object(image, max_size_to_erase, min_size=0):
    """
    Erase objects in the image whose pixel size is between min_size and max_size_to_erase.

    :param image: 2D numpy.ndarray - Input image
    :param max_size_to_erase: int - Maximum size of the object to erase
    :param min_size: int - Minimum size of the object to erase (default: 0)
    :return: 2D numpy.ndarray - Image with objects erased
    """
    num_object, label, stats, centroids = cc_stat(image)
    for i in range(1, num_object):
        object_size = stats[i, cv.CC_STAT_AREA]
        if object_size < max_size_to_erase and object_size > min_size:
            image[label == i] = 0
    return image


def binary_image(image, thresh, inv=False):
    """
    Convert the input image to a binary or inverse binary image based on global threshold method.
    :param image: 2D\3D numpy.ndarray uint16 - Input image
    :param thresh: float - Threshold value
    :param inv: int - Inverse flag (0 for binary, 1 for inverse binary) (default: 0)
    :return: 2D\3D numpy.ndarray - Binary or inverse binary image
    """
    bin_image = np.copy(image)
    if inv is False:
        bin_image[image <= thresh] = 0
        bin_image[image > thresh] = 2 ** 16 - 1
    else:
        bin_image[image <= thresh] = 2 ** 16 - 1
        bin_image[image > thresh] = 0
    return np.uint8(bin_image)


def area_object(image):
    """
    Return the area (number of pixels) of the objects in the image without the background object.
    :param image: 2D numpy.ndarray or 2D matrix - Input image
    :return: list of int - Maximum pixel area of the objects
    """
    stats = cc_stat(image)[2]
    if len(stats) > 1:
        return stats[1:, cv.CC_STAT_AREA]
    else:
        return [0]


def top_pix(image):
    """
    Find the indexes of the upper pixel in the image.
    :param image: 2D numpy.ndarray - Input image
    :return: tuple of int - Indexes of the upper pixel (u, v)
    """
    n, m = image.shape
    for u in range(n):
        for v in range(m):
            if image[u, v] != 0:
                return u, v
    return (-1, -1)



def hu(image, slope=1, intercept=-1024):
    """
    Convert the image to Hounsfield units using slope and intercept.

    :param image: 2D\3D numpy.ndarray - Input image
    :param slope: float - Slope value (default: 1)
    :param intercept: float - Intercept value (default: -1024)
    :return: 2D\3D numpy.ndarray - Image in Hounsfield units
    """
    image = image * slope + intercept
    return image


def normalize(image):
    """
    Windowing the image and rescaling pixel values to the range of [0, 255].
    :param image: 2D numpy.ndarray - Input image
    :return: 2D numpy.ndarray - Windowed image
    """
    # windowing range [-800,1000]
    MIN_BOUND = -800
    MAX_BOUND = 1000

    # rescaling
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1
    image[image < 0] = 0
    image = image * 255
    image = np.rint(image)
    return np.uint8(image)


def gray_to_color(image, hu_val=True, slope=1, intercept=-1024 ):
    """
    Rescale the image to Hounsfield units, window the image, and transform it to a color image.

    :param image: 2D\3D numpy.ndarray - Input image
    :param hu_val: bool - True if the image is in Hounsifeld unit values
    :param slope: float - Slope value (default: 1)
    :param intercept: float - Intercept value (default: -1024)
    :return: 3D numpy.ndarray - Color image (3 values per pixel)
    """
    if hu_val is False:
        image = hu(image, slope, intercept)

    normalized_image = normalize(image)
    if image.ndim == 2:
        color_image = np_to_color(normalized_image)
    else:
        color_image = np_3D_to_color(normalized_image)
    return color_image



def np_to_color(image):
    """
    Transform a gray scale image (1 value per pixel) to an RGB image (3 values per pixel).

    :param image: 2D numpy.ndarray - Input image
    :return: 3D numpy.ndarray - RGB image (3 values per pixel)
    """
    return np.stack((image, image, image), axis=2)


def np_3D_to_color(images):
    """
    Transform a 3D numpy array (gray scale images) to a 4D numpy array (RGB images).

    :param images: 3D numpy.ndarray - Input images
    :return: 4D numpy.ndarray - RGB images (3 values per pixel)
    """
    images = np.array(images)
    return np.stack((images, images, images), axis=3)


def axial_to_sagittal(image):
    """
    Convert axial images to sagittal images.

    :param image: 3D numpy.ndarray - Axial images
    :return: 3D numpy.ndarray - Sagittal images
    """
    return np.swapaxes(image, axis1=0, axis2=2)


def axial_to_coronal(image):
    """
    Convert axial images to coronal images.
    :param image: 3D numpy.ndarray - Axial images
    :return: 3D numpy.ndarray - Coronal images
    """
    return np.swapaxes(image, axis1=0, axis2=1)

# this function take only the biggest object of the nasal airway
def remove_unconnected_objects(images):
    # take only the big object from the 3D image
    labels_out = cc3d.largest_k(np.array(images), k=1, connectivity=18, delta=0)
    cc_airway = np.array(images)
    cc_airway[labels_out != 1] = 0
    return cc_airway


#
def concat(df1, df2):
    """
    Concatenate two DataFrames after ensuring their columns have the same data types.

    :param df1: The first pandas DataFrame.
    :param df2: The second pandas DataFrame.
    :return: Concatenated DataFrame with harmonized data types.
    """

    # Convert the columns of df1 to the data types of df2
    df1_converted = df1.astype(df2.dtypes)

    # Convert the columns of df2 to the data types of df1
    df2_converted = df2.astype(df1.dtypes)

    # Concatenate the two DataFrames with harmonized data types
    return pd.concat([df1_converted, df2_converted],ignore_index=True)
