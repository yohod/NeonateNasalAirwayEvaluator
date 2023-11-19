import numpy as np
import cv2 as cv
import cc3d
import closing_nares as closn
import usefull_function as uf


def seg_nasal_airway(slices, all_head, nose_images, edge_index, end_open_nose_index, naso_index, end_nasopharynx,
                     image_type, global_thresh=-400, local_thresh=(-400,-125)):
    """
    Segment the nasal airway in medical image data.

    :Parameters:
    slices (list): List of image slices.
    all_head (list): List of all head images from the ROI slices.
    nose_images (list): List of cut ROI images.
    edge_index (int): Index of the edge of the nose.
    end_open_nose_index (int): Index marking the end of the open nostril.
    naso_index (int): Index indicating one of the slices of the nasopharynx.
    end_nasopharynx (int): Index marking the end of the nasopharynx.
    image_type (int): Type of image orientation.
    global_thresh: the HU value for the global threshold method
    # default: -400. air can vary in different CT scans. This was the better HU value on our cases research
    # Trade of between including unrelated pixels, not detecting all the narrow airways
    local_thresh (tuple of float): Min, MAx threshold value for the local thresholding method.
    default: min = -400 HU. the global threshold, max = -125. soft tissues and fat values somtimes are -100 HU

    :Returns:
    tuple: A tuple containing segmented slices and related data.

    :Details:
    - Copy the ROI images for further processing.
    - Perform nostril closing on the ROI images.
    - Calculate the pixel width from the PixelSpacing of the first slice.
    - Invert the airway to white and other regions to black.
    - Remove unwanted objects like oropharynx and unconnected regions below the nasopharynx.
    - Improve the segmented images using local thresholding.
    - Adjust for any black images at the start of the sequence.
    - Update the data based on the removal of black images.
    - Return the segmented slices and related data.
    """
    CLOSING_NOSTRIL_THRESH = -325 # work well with this threshold

    # After comparing several values,
    # the radiologist chose -325 HU as the nominal global threshold value.
    # however, because the use of a local threshold value method as well,
    # a lower global threshold value was chosen (-400 HU)
    # that is closer to the one mentioned in the literature as nominal at adult nasal airway (around -450 HU)
    # (for example: https://iopscience.iop.org/article/10.1088/2057-1976/aac6af, DOI 10.1088/2057-1976/aac6af)

    roi_images = np.copy(nose_images)
    close_roi_images, open_nostril_index = closn.closing_nostril(roi_images,  CLOSING_NOSTRIL_THRESH , edge_index, end_open_nose_index)
    pixel_width = slices[0].PixelSpacing[0]

    # remove surrounding air, mouse and nose cavity
    inv_images = inverse_airway_to_white_without_background(uf.binary_image(close_roi_images, global_thresh))
    seg_images, max_index = remove_mouse_and_oropharynx(inv_images, edge_index, end_open_nose_index, naso_index,
                                                        end_nasopharynx, image_type)

    # using local threshold to include narrow regions
    min_thresh = local_thresh[0]
    max_thresh = local_thresh[1]
    fix_seg_images = seg_correction(close_roi_images, seg_images, max_index, naso_index, end_nasopharynx,
                                    open_nostril_index, end_open_nose_index, pixel_width,
                                    min_thresh=min_thresh, max_thresh=max_thresh)

    # Removing any initial black images
    for start in range(5):
        if np.count_nonzero(fix_seg_images[start]) > 0:
            break

    if start > 0:
        slices = slices[start - 1:]
        nose_images = nose_images[start - 1:]
        all_head = all_head[start - 1:]
        seg_images = seg_images[start - 1:]
        fix_seg_images = fix_seg_images[start - 1:]
        open_nostril_index -= (start - 1)
        edge_index -= (start - 1)

    return slices, all_head, nose_images, seg_images, fix_seg_images, open_nostril_index, edge_index


# A utility function designed to ensure that each unique object number is encountered only once.
def extract_object_numbers(input_list):
    """
    Create a list containing all unique object numbers from the input_list.

    Parameters:
    input_list (list): The input list containing object numbers.

    Returns:
    list: A new list containing unique object numbers.
    """
    unique_numbers = []
    for sub_list in input_list:
        for number in sub_list:
            if number not in unique_numbers:
                unique_numbers.append(number)
    return unique_numbers


# transform all the air in the background to black
def transform_background_to_black(image):
    """
    Convert the background air of the image to black.

    Parameters:
    image (2D numpy array): The input image.

    Returns:
    2D numpy array: The modified image with the background set to black.
    """
    label = uf.cc_stat(image)[1]
    for obj in extract_object_numbers([label[0, :], label[:50, 0], label[:50, -1]]):
        image[label == obj] = 0
    return image


# inverse the image - the airway become white, and all the reset black
def inverse_airway_to_white_without_background(images):
    """inverse the image - the airway become white, and all the reset black """


    images = [(255 - np.uint8(image)) for image in images]
    # transform the front background from white to black
    images = [transform_background_to_black(image) for image in images]
    return images




def not_in(label, top_boundary_index):
    """
    Create a list of objects that are not connected and are located below the top boundary index.
    This is useful for removing unwanted objects that are not part of the nasal airway.

    :Parameters:
     label: 3D numpy array containing labels of connected components in the image.
     top_boundary_index: Index marking the top boundary slice.

    :Returns: A list of label numbers representing objects to be removed.
    """
    max_num = np.max(label)
    max_in_first = np.max(label[:top_boundary_index + 2])
    not_in_list = [*range(max_in_first + 1, max_num + 1)]

    # Ensure not to erase the nasopharynx
    # The nasopharynx is surely in the last slice (-1) and -11 rows from the end.
    nasopharynx_list = label[-1, -15:, :].flatten()
    for el in nasopharynx_list:
        if el in not_in_list:
            not_in_list.remove(el)

    return not_in_list


# clean image from the mouse and other object that is not connected and lower from the longest object
# may be improved (mouse is in the front)

def remove_mouse_and_oropharynx(images, edge_index, end_open_nose_index, naso_index, end_nasopharynx, image_type):
    """
    Remove unwanted objects like the oropharynx and unconnected regions below the nasopharynx.

    :Parameters:
    images (list of 2D numpy arrays): A list of image slices.
    edge_index (int): Index of the edge of the nose.
    end_open_nose_index (int): Index marking the end of the open nostril.
    naso_index (int): Index indicating one of the slice of the nasopharynx.
    end_nasopharynx (int): Index marking the end of the nasopharynx.
    image_type (int): Type of image orientation.

    :Returns:
    tuple of 2D numpy arrays and int: A tuple containing cleaned images and
     the index describe the top boundary slice.

    :Details:
    - The topmost slice of the nasopharynx below which objects are searched for deletion is defined.
    - Depending on the image type, different strategies are applied to locate the nasopharynx:
      - Image type 1 or 3: Searches for the connection of the airway to the nasopharynx in images with long objects.
      - Image type 2: When the head is tilted upwards, the nasopharynx begins after the long object.
    - The oropharynx is removed in cases where it includes slices below the end of the nasopharynx.
    - Objects to be erased are identified based on their label and their location in the image.
    - Objects that need to be preserved, such as the nasopharynx, are not removed.
    """

    n, m = images[0].shape

    # Define the topmost slice of the nasopharynx below which objects are searched for deletion
    top_boundary_index = 20
    if image_type == 1 or image_type == 3:
        # Search for the connection of the airway to the nasopharynx (images with long objects).
        index = 20
        max_len = 0
        for image in images[20:end_open_nose_index]:
            stat = uf.cc_stat(image)[2] # take only the data about the objects
            for length in stat[1:, cv.CC_STAT_HEIGHT]:
                if length >= max_len:
                    max_len = length
                    top_boundary_index = index
            index += 1

    elif image_type == 2:
        # When the head is tilted upwards, the nasopharnx begins after the long object.
        top_boundary_index = naso_index
        for i in range(naso_index, end_open_nose_index, -1):
            image = images[i]
            num_of_objects = uf.cc_stat(image[:n // 3, :])[0]
            if num_of_objects < 2:
                break
            top_boundary_index -= 1

    # Remove the oropharynx
    # In cases that include slices of the oropharynx below the end of the nasopharynx
    if end_open_nose_index > end_nasopharynx:
        for image in images[end_nasopharynx:]:
            image[n // 2:, :] = 0

    cleaning = np.array(images)
    cleaning[:, -10:, :] = 0
    label_out = cc3d.connected_components(cleaning)

    # Create the list of objects to erase
    obj_to_erase = not_in(label_out, top_boundary_index) + ears_cavitis_object(label_out)

    # Erase objects
    for num in obj_to_erase:
        cleaning[label_out == num] = 0

    return cleaning, top_boundary_index

def ears_cavitis_object(label):
    """
    Find the label numbers of ear cavities, objects located at the sides of the images and small.
    :Parameter: label: 3D numpy array containing labels of connected components in the image.
    :Returns: A list of label numbers representing ear cavities.
    """
    n, m, k = label.shape
    # Extract label numbers from the sides of the images
    object_numbers = list(label[:,-m//4:,-4:].flatten() + label[:,-m//4:,:4].flatten())
    ears_object = []

    # Check for small objects and add them to the list
    if np.amax(object_numbers) > 0:
        for i in object_numbers:
            if i == 0:
                continue
            elif np.count_nonzero(label == i) < 1000 and i not in ears_object:
                ears_object.append(i)

    return ears_object




def seg_correction(close_roi_images, seg_images, max_index, naso_index, end_nasopharynx, open_nostril_index, end_open_nose_index, pixel_width, min_thresh=-400, max_thresh = -125 ):
    """
    Improve the segmented images by applying local thresholding to enlarge the segmented images. The goal is to include narrow regions that weren't detected in the global threshold method.

    :Parameters:
    close_roi_images (list of 2D numpy arrays): Closed nostril images.
    seg_images (list of 2D numpy arrays): Segmented airway images.
    max_index (int): Maximum index.
    naso_index (int): Nasopharynx index.
    end_nasopharynx (int): End of nasopharynx index.
    open_nostril_index (int): Open nostril index.
    end_open_nose_index (int): End of open nose index.
    min_thresh (float): Threshold value for the local thresholding.
    default: = -400 HU value. the global threshold value
    max_thresh Maximum threshold value for local thresholding,
    default: = -125 HU value. soft tissue and fat can have -100 HU value
    pixels with a value bigger to this value and smaller than -125 HU checked
    pixel_width (float): Width of a pixel in the images.

    :Returns:
    list of 2D numpy arrays: Corrected segmented images.

    :Details:
    - Connectivity: 18 (indicates the connectivity for connected components)
    - l, n, m: Dimensions of the segmented images
    - end: The boundaries for the region of improvement based on various indices
    - row_start and row_end: Define the rows of interest for improvement (coronal direction)
    - area_factor: Determines the size threshold for small object removal
    - second_seg: Segmented images created using local thresholding
    - fix_seg_images: The segmented images with small objects removed and improved local thresholding
    - region_growing: A function used to expand the segmented images based on local thresholding
    """

    connectivity = 18
    l, n, m = np.shape(seg_images)

    # The boundaries for the region of improvement
    end = max(max_index + 5, naso_index)
    if open_nostril_index == end_open_nose_index:
        end = end_nasopharynx
    row_start = 16 * n // 100
    row_end = -int(0.4 * n)

    # Remove small objects from about 1.5 mm^2
    if pixel_width > 0.33:
        area_factor = 1
    elif pixel_width > 0.26:
        area_factor = 2.5
    else:
        area_factor = 4
    fix_seg_images = cc3d.dust(np.array(seg_images), 5 * area_factor, connectivity=connectivity)

    # Enlarging the first segmentation using local threshold.
    second_seg = np.array([local_threshold(image, min_thresh=min_thresh, max_thresh=max_thresh)
                           for image in close_roi_images])
    fix_seg_images[:end, row_start:row_end, :] = region_growing(fix_seg_images,
                second_seg, condition=True, begin_row=row_start, end_row=row_end, end_slice=end)

    # Remove small objects in the improved segmented images
    for image in fix_seg_images:
        num, label, stat, cent = uf.cc_stat(image)
        for i in range(1, num):
            area = stat[i, cv.CC_STAT_AREA]
            y_cent, x_cent = cent[i]
            if area < 2 * area_factor:
                image[label == i] = 0

    return fix_seg_images

def region_growing(first_seg, second_seg, condition=False, begin_row=0,end_row= 0,start_slice=0,end_slice = 0):
    """
    Applies region growing to enhance the segmented images.
    if condition is True the disconnected objects are growing only if the big part of the pixels
    are in the center.

    :Parameters:
    first_seg (3D numpy array): The original segmented images
    second_seg (3D numpy array): The second segmented images for enlarging the first one.
    condition (bool): A condition for processing.

    The boundaries for applay the function
    begin_row (int): Starting row for processing. (coronal direction)
    end_row (int): Ending row for processing.
    start_slice (int): Starting slice for processing.(axial direction)
    end_slice (int): Ending slice for processing.

    :Returns:
        3D numpy array: Enhanced segmented images.
    """

    if end_row == 0:
        end_row = first_seg.shape[1]
    if end_slice == 0:
        end_slice = first_seg.shape[0]

    connectivity = 18 # To prevent objects that are almost not connected from being considered the main object

    # Map the objects of the initial segmentation (segmentation with a global threshold value)
    label_first_seg = cc3d.connected_components(np.array(first_seg[start_slice:end_slice
                ,begin_row:end_row, :]), connectivity=connectivity)
    num_of_objects = np.amax(label_first_seg) + 1

    #Map the objects of the second segmentation (after also using a local threshold value)
    label_second_seg = cc3d.connected_components(second_seg[start_slice:end_slice
                , begin_row:end_row, :], connectivity=connectivity)

    # find the label number of the main object of the airway
    connected_num = cc3d.largest_k(label_second_seg, k=1)
    x, y, z = tuple(np.transpose(np.nonzero(connected_num == 1))[0])
    connected_num = label_second_seg[x, y, z]

    growing_matrix = np.copy(first_seg[start_slice:end_slice, begin_row:end_row, :])

    # The loop running over all the primary objects
    # Only increases these primary areas of the first segmentation.
    # Objects that were not connected in the first segmentation
    # and grew significantly are not added

    for i in range(1, num_of_objects):

        # Finds the coordinates of the first pixel location of the object
        x, y, z = tuple(np.transpose(np.nonzero(label_first_seg == i))[0])
        #  Identifies the object number at this location in the second segmentation
        obj_num = label_second_seg[x, y, z]

        if obj_num == 0: #  If the object num indicates the background, ignore
            continue

        if obj_num != connected_num:
            # for some_cases with a big effect on side of image
            # Only enlarge objects that are connected
            # or objects that were not connected but the pixel addition is mainly in the center of the image,
            # i.e., the airway.
            # If the number of pixels on the sides is relatively large, it indicates areas that are not airway,
            # and therefore do not enlarge the area to include them.

            if condition is True:
                difference = np.count_nonzero(label_second_seg == obj_num) - np.count_nonzero(label_first_seg == i)
                image_col = len(label_first_seg[0, 0, :])
                ratio_in_top_left = np.count_nonzero(label_second_seg[:, :, :int(0.3*image_col)] == obj_num) / difference
                ratio_in_top_right = np.count_nonzero(label_second_seg[:, :, -int(0.3*image_col):] == obj_num) / difference
                if ratio_in_top_right > 0.5 or ratio_in_top_left > 0.5:
                        # If at least half of the added pixels is in the center of the image
                         continue
        growing_matrix[label_second_seg == obj_num] = 255

    return growing_matrix

def local_threshold(image, min_thresh, max_thresh):
    """
    Apply local thresholding to include narrow region of airway.
    to pixel between min_thersh and -125HU

    :param: image (2D numpy array): The input image.
    :param: min_thresh (int): The minimum threshold value.
     default: = -400 HU value. the global threshold value
    :param: max_thresh Maximum threshold value for contrast enhancement.
     default: = -125 HU value. soft tissue and fat can have -100 HU value

    :Returns:
    2D numpy array: The image with enhanced contrast, focusing on the airway regions.
    """
    n, m = image.shape
    kernel_image = np.copy(image)


    contrast_matrix = np.copy(image)
    contrast_matrix[image < min_thresh] = 255
    contrast_matrix[image > max_thresh] = 0

    for i in range(2, n - 9):
        for j in range(2, m - 2):
            if contrast_matrix[i, j] == 0 or contrast_matrix[i, j] == 255:
                continue
            else:
                kernel = kernel_image[i - 1:i + 2, j - 1:j + 2]
                local_thresh = np.mean(
                    kernel[kernel > min_thresh - 100])
                if contrast_matrix[i, j] < local_thresh:
                    contrast_matrix[i, j] = 255
                else:
                    contrast_matrix[i, j] = 0

    # Erase background
    _, label, stat, _ = uf.cc_stat(contrast_matrix)
    contrast_matrix[label == label[0, 0]] = 0
    contrast_matrix[label == label[0, -1]] = 0

    return np.uint8(contrast_matrix)


