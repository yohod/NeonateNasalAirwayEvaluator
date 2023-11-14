import os
import pydicom as dicom  # DICOM(Digital Imaging and COmmunications in Medicine)
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import usefull_function as uf


# 1.1 load and sort the dicom images from the top to bottom
def load_dicom(path=""):
    """load ct images from path and sort it from superior (head) to inferior (neak)
    :param path: = the path for the CT dicom files. defualt is empty str, give posibilty to input it.
    :type path: str
    :return: sort list of  the CT slices
    :rtype: list of dicom object """
    if path == "":
        path = input("enter the path of the CT images")
    ct_images = os.listdir(path)
    # This method returns a list containing the names of the entries in the directory given by path
    slices = [dicom.dcmread(path + '/' + i, force=True) for i in ct_images]
    # Read and return a dataset not in accordance with the DICOM File Format:
    # dcmread / read_file two function for load dicom files

    # sorting ct images according to their position it's more precise. the positive direction is to the superior of the body
    # if not exist, sorting ct images by their InstanceNumber,which is the slice index.
    if hasattr(slices[1], "ImagePositionPatient"):
        slices = sorted(slices, reverse=True, key=lambda x: x.ImagePositionPatient[2])
        sorted_by_position = True

    elif hasattr(slices[1], "InstanceNumber"):
        slices = sorted(slices, key=lambda x: x.InstanceNumber)
        sorted_by_position = False
    else:
        print("cant be sorted")

    # remove MIP images, if they exist in the beginning of the series
    for i in range(0,3):
        if slices[i].PixelSpacing[0] != slices[i+1].PixelSpacing[0]:
            continue
        else:
            slices = slices[i:]
            break

    images = [i.pixel_array for i in slices]

    # calculate slice thickness using the axial direction differences in the image position.
    # it's more accurate to use the ImagePositionPatient attribute from use SliceThickness attribute
    if sorted_by_position is True:
        slice_thickness = []
        for i in range(len(slices) - 1):
            slice_thickness.append(
                abs(round(slices[i + 1].ImagePositionPatient[2] - slices[i].ImagePositionPatient[2], 3)))
        slice_thickness = round(np.mean(slice_thickness), 3)
        reverse = slices[0].InstanceNumber > slices[1].InstanceNumber
    else:
        # when sorting by the instance number sometimes need to reverse the order to be from superior to inferior.
        slices, images,reverse = reverse_order(slices,images)
        slice_thickness = slices[1].SliceThickness

    # convert image values to Hounsfield units (HU)
    slope = slices[1].RescaleSlope
    intercept = slices[1].RescaleIntercept
    images = uf.hu(np.array(images), slope=slope,intercept=intercept)

    # spacing on (axial,coronal,sagittal) direction
    spacing_xy = slices[1].PixelSpacing
    spacing = (slice_thickness, spacing_xy[0], spacing_xy[1])

    return slices, images, spacing, reverse

# 1.2 reorder slices to begin from head's top to bottom if it was in reverse order


def reverse_order(slices, images):
    """reorder ct slices from head(superior) to neck(inferior) by comparing
    the number of air objects in the begin and end of the list of images
    :type slices: list of dicom
    :rtype: list of dicom"""

    start_counter = 0
    end_counter = 0
    list_length = images.shape[0]
    for i in range(1, list_length // 2):
        begin_image = images[i]
        start_counter += count_obj(begin_image)
        end_image = images[-i]
        end_counter += count_obj(end_image)

        if start_counter - 10 >= end_counter:
            # print (i,"r",start_counter, end_counter)
            reverse = True
            return slices[::-1], images[::-1], reverse

        elif start_counter <= end_counter - 10:
            # print (i,"V", start_counter, end_counter)
            reverse = False
            return slices, images, reverse

    # don't need this extra lines
    if start_counter > end_counter:
        reverse = True
        return slices[::-1], images[::-1], reverse

    else:
        reverse = False
        return slices, images, reverse


# help function for "reverse_order" function
def count_obj(image):
    """count the number of the air objects in the image which is bigger than 25 pixels
    :type image: 2d np.ndarray
    :return: the number of air objects in the image
    :rtype: int"""
    binary_img = uf.binary_image(image, thresh=-525, inv= 1) # -525
    binary_img = uf.erase_object(binary_img, max_size_to_erase=25)  # erase small object
    num_of_objects = uf.cc_stat(binary_img)[0] # count the number of air object
    return num_of_objects



# 1.3  rotate the image if the head is tilted
# the angle rotation for each case in my research
# To save time, and avoid having to check the correct angle every time.
# the angle is in the XY plane
def angle(i):
    angles = {'0':2,'1': 0, '2': 0, '3': 45, '4': -5, '5': -78, '6': 0, '7': -50, '8': 0, '9': -10, '10': 0, '11': 0,
              '12': 57, '13': 0, '14': -39, '15': -65, '16': -18, '17': 0, '18': 0, '19': -95, '20': -75, '21': -7,
              '22': -80, '23': -5, '24': 56, '25': -4, '26': 33, '27': 33, '28': -35}
    return angles[str(i)]


# rotating automatically the images in my research
def fix_angle(images, rot_angle):
    """rotate ct images to redirect the nose to top of the image
    :type images: numpy 3D array of head images
    :type rot_angle: int or float
    :rtype: 3D numpy.ndarray"""
    fix_images = np.copy(images)
    list_length = images.shape[0]
    for i in range(list_length):
        image_center = tuple(np.array(images[i].shape[1::-1]) / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, rot_angle, 1.0)
        fix_images[i] = np.array(cv.warpAffine(images[i], rot_mat, images[i].shape[1::-1], flags=cv.INTER_LINEAR))
    return fix_images


# a semi-automate method for rotate the images
def corect_angle(images):
    """ show image and rotate with user input to direct the nose in the up direction.
    :type images: list of 2d numpy.ndarray
    :rtype: list of 2d numpy.ndarray"""
    mid_index = images.shape[0] // 2
    plt.imshow(images[mid_index], cmap=plt.cm.gray)
    plt.show()
    angle = int(input("enter an angle to rotate int: "))
    while (angle != 0):
        images = fix_angle(images, angle)
        plt.imshow(images[mid_index], cmap=plt.cm.gray)
        plt.show()
        angle = int(input("enter an angle to rotate int: "))

    return images


# 1.4 remove slices with lack of data, which the image is cut significantly
def remove_slices(slices, images):
    """ remove the slices that are black or which a significant cutting of data
    return: updated slices and images
    """
    index_begin = 0
    # remove images that the head is small pixels are less than 20,000
    for image in images:
        if np.count_nonzero(uf.binary_image(image, thresh=-525)) < 20000:
            index_begin += 1
        else:
            break

    index_end = images.shape[0]
    for image in images[::-1]:
        if np.count_nonzero(uf.binary_image(image, thresh=-525)) < 20000:
            index_end -= 1
        else:
            break

    for index in range(index_end - 1, index_begin, -1):
        # print(index ,end = ": ")
        _, _, stat_last, _ = uf.cc_stat(uf.binary_image(images[index], thresh=-525))
        area_last = 0
        for area in stat_last[1:, cv.CC_STAT_AREA]:
            if area_last < area:
                area_last = area
        _, _, stat_before, _ = uf.cc_stat(uf.binary_image(images[index - 1], thresh=-525))
        area_before = 0
        for area in stat_before[1:, cv.CC_STAT_AREA]:
            if area_before < area:
                area_before = area
        #
        if area_last + 3500 > area_before:
            break
        else:
            index_end -= 1
    images = images[index_begin:index_end]
    slices = slices[index_begin:index_end]

    return slices, images




# 1.5 preprocessing
def preprocessing(path, my_research=True):
    """upload dicom files from path and preprocess the slices and images
    :type path: str
    :rtyp
    :return: list of dicom, 3D numpy.ndarray of images, spacing of the image in (axial,coronal,sagittal))
    # reverse and image_for_sagittal is parameters for the sagittal presentation"""
    slices,images, spacing ,reverse = load_dicom(path)


    # take from path the case number
    if my_research is True:
        if path[-2:].isdigit():
            case_num = path[-2:]
        elif path[-1].isdigit():
            case_num = path[-1]
        images = fix_angle(images, angle(case_num))
        images_for_sagittal = np.copy(images)
    else:
        images = corect_angle(images)
        images_for_sagittal = np.copy(images)

    slices, images = remove_slices(slices, images)

    return slices, images_for_sagittal, images, spacing, reverse
