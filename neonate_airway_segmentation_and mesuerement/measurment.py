import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv
import cc3d
from skimage.measure import mesh_surface_area, marching_cubes
import math
import usefull_function as uf
import volume
import cross_sectional as cs



# Find approximately the pyriform aperture (PA).
# Simple method: Use the first bone in the axial edge_index slice.
# Complicated method: Take into account the head angle in the scan as well.

def find_aperture(roi_images, edge_index, image_type, simple_method=True):
    """
    Find the pyriform aperture (PA) in neonatal nasal images.

    :param roi_images: List of region of interest images.
    :param edge_index: Index of the image of the nose tip.
    :param image_type: Scan angle - 1: normal, 2: head angled up, 3: head angled down.
    :param simple_method: Flag for using the simple method. Default is True.

    :return: Coronal index of the pyriform aperture.

    Note:
        The Pyriform aperture is an anatomical feature in the nasal region.
    """

    if simple_method is True:
        index = edge_index
    else:
        # A method to try to correct the accuracy of the PA according to
        # the state of the scan (scan angle - 1: normal, 2: head angled up, 3: head angled down)
        if edge_index - 10 >= 25:
            index = edge_index - 10
        elif edge_index < 25:
            index = edge_index
        else:
            index = 25
        if image_type == 2:
            index = edge_index

    # Binary image processing
    image = uf.binary_image(roi_images[index], 200)

    # Segmenting the bone. Bone value can vary from 200 HU.
    # Neonate bones sometimes not fully calcified.
    image = uf.erase_object(image, 200)
    num, label, stat, cent = uf.cc_stat(image)

    # Remove small objects based on top pixel position
    for i in range(1, num):
        if stat[i, cv.CC_STAT_TOP] < 20:
            image[label == i] = 0

    # Get the coronal index of the Pyriform aperture
    coronal_pa_index = uf.top_pix(image)[0]

    # Visualization check
    if False:  # Set to True for visualization
        print(image_type)
        image = np.stack((np.uint8(image),) * 3, axis=-1)
        image[coronal_pa_index - 1:coronal_pa_index + 1, :] = (255, 0, 0)
        image2 = uf.gray_to_color(roi_images[edge_index], 1, -1024)
        image2[coronal_pa_index - 1:coronal_pa_index + 1, :] = (255, 0, 0)
        for img in [image2, image]:
            plt.imshow(img)
            plt.yticks([])
            plt.xticks([])
            plt.show()

    return coronal_pa_index




# Find the posterior nostril coronal index.
# In the posterior nostrils, the two nasal passages connect to the nasopharynx.
# The nasopharynx is wide, and there is a coronal slice where it splits into two canals,
# and its width is significantly reduced (at least 1.5 times).
# This algorithm may not work accurately when there is a tube passing through the pharynx.

def find_choana(seg_images, roi_images):
    """
    Find the coronal index of the posterior nostril,
    specifically where the nasopharynx splits into two canals with reduced width.

    :param seg_images: List of segmented images.
    :param roi_images: List of region of interest images.
    :return: Coronal index of the posterior nostril.

    Note:
        The posterior nostrils connect to the nasopharynx,
        and the algorithm considers the width reduction from the nasopharynx.
    """

    coronal_images = uf.axial_to_coronal(seg_images)

    n = len(coronal_images)
    width = -1
    choana_index = n - 40  # Initial index. Coronal index n-40, assuming it is on the nasopharynx.

    # Search for the last time when the width of the widest object
    # becomes 1.5 times wider than the previous coronal slices.

    for index in range(choana_index, n // 2, -1):
        image = coronal_images[index]
        num, label, stat, _ = uf.cc_stat(image)
        if num > 1:
            current_width = np.amax(stat[1:, cv.CC_STAT_WIDTH])
        else:
            current_width = 0
            # two_sided_atresia = True
        if current_width * 1.5 < width:
            choana_index = index
            break
        width = current_width

    # If it didn't find any slice where the width became 1.5 times smaller than before,
    # this means that there is interference.
    if False:  # Visualization check
        for img in uf.axial_to_coronal(roi_images)[choana_index - 1: choana_index + 2]:
            img = uf.gray_to_color(img)
            plt.imshow(img)
            plt.show()
            print(width, current_width)

    return choana_index


# Gets a map of the respiratory tract from the beginning of the nostrils to the nasopharynx,
# and identifies which object is the right canal and which is the left canal.

def find_side_cc_label_number(cc_label):
    """
    Get a map of the respiratory tract and identify the right and left canals.

    :param cc_label: 3D array representing connected components labeling of the respiratory tract.

    :return:
        cc_label (numpy.ndarray): Updated connected components labeling array.
        left_num (int): Label number for the left canal.
        right_num (int): Label number for the right canal.

    Note:
        The function examines the connected components labeling to determine which object represents the left nostril
        and which represents the right nostril in the respiratory tract.
    """

    num_of_objects = np.amax(cc_label)
    _, _, p = cc_label.shape

    # Get the first pixel row in each side
    if num_of_objects >= 2:
        first_index = np.where(cc_label == 1)[2][0]
        second_index = np.where(cc_label == 2)[2][0]

        # Decide which object is the left nostril and which is the right
        if first_index < second_index:  # (cc num 1 is the left nostril)
            left_num = 1
            right_num = 2
        else:  # (cc num 2 is the left nostril)
            left_num = 2
            right_num = 1

        if num_of_objects > 2:
            for i in range(3, num_of_objects + 1):
                first_obj = np.where(cc_label == i)[2][0]
                if first_obj < p // 2:
                    cc_label[cc_label == i] = left_num
                else:
                    cc_label[cc_label == i] = right_num

    elif num_of_objects == 1:
        left_pixels = np.count_nonzero(cc_label[:, :, :p // 2])
        right_pixels = np.count_nonzero(cc_label[:, :, p // 2:])
        if left_pixels > right_pixels:
            left_num = 1
            right_num = -1
        else:
            right_num = 1
            left_num = -1

    elif num_of_objects == 0:
        left_num = -1
        right_num = -1

    return cc_label, left_num, right_num


# Divides the unconnected objects by channel sides.

def find_side_notcc_label_numbers(notcc_label):
    """
    Divide unconnected objects by channel sides.

    :param notcc_label: 3D array representing labeling of unconnected objects.

    :return:
        notcc_left_nums (list): List of label numbers for objects on the left side.
        notcc_right_nums (list): List of label numbers for objects on the right side.

    Note:
        The function counts the pixels of each unconnected object in the left and right sides of the channel
        and categorizes them based on the side with more pixels.
    """

    num_of_objects = np.amax(notcc_label)
    _, _, p = notcc_label.shape
    notcc_left_nums = []
    notcc_right_nums = []

    for i in range(1, num_of_objects + 1):
        left_pixels = np.count_nonzero(notcc_label[:, :, :p // 2] == i)
        right_pixels = np.count_nonzero(notcc_label[:, :, p // 2:] == i)
        if left_pixels > right_pixels:
            notcc_left_nums.append(i)
        else:
            notcc_right_nums.append(i)

    return notcc_left_nums, notcc_right_nums

# Roughly estimate where the tip of the lower concha is.
# Define two points for each side. Point 1 in the PA (Pyriform Aperture) area,
# and is the middle of the height of the airways in this area.
# (+ correction on each side related to point 2)
# and point 2 in the CH (Choanae) area, which is characterized
# by the upper edge of the posterior nostrils on each side + bias.
# The goal of 2 points to create a straight line with a slope and
# thus cope with the angle of the scan.
# For the same reason (scanning angle) two points are defined for each side.

def finding_inferior_concha(seg_images, pa_index, ch_index, type):
    """
    Roughly estimate the tip of the lower concha.

    :param seg_images: Segmented nasal airway, list of 2D numpy arrays.
    :param pa_index: Coronal index of the Pyriform Aperture.
    :param ch_index: Coronal index of the Choanae.
    :param type: Scan type - 1: normal, 2: head angled up, 3: head angled down.

    :return:
        pa_mid (tuple): Axial indices of the inferior concha in the Pyriform Aperture coronal slice.
        ch_mid (tuple): Axial indices of the inferior concha in the Choana coronal slice.
    """
    ch_image = seg_images[:, ch_index + 1, :]
    n, m = ch_image.shape
    num, label, stat, cent = uf.cc_stat(ch_image)

    # Find the left and right high pixel =  approximation of the inferior concha
    if num > 1:  # For added certainty
        pharynx_num = np.argmax(stat[1:, cv.CC_STAT_WIDTH]) + 1
        label[label != pharynx_num] = 0
        mid_pix = stat[pharynx_num, cv.CC_STAT_LEFT] + stat[pharynx_num, cv.CC_STAT_WIDTH] // 2
        left_stat = uf.cc_stat(label[:, :mid_pix])[2]
        right_stat = uf.cc_stat(label[:, mid_pix:])[2]
        left_flag = False
        right_flag = False
        l_height = left_stat[1, cv.CC_STAT_HEIGHT]
        r_height = right_stat[1, cv.CC_STAT_HEIGHT]
        l_down = left_stat[1, cv.CC_STAT_TOP] + l_height
        r_down = right_stat[1, cv.CC_STAT_TOP] + r_height

        if l_height >= 15:
            left_mid_ch = l_down - 13
            left_flag = True

        elif left_stat[1, cv.CC_STAT_TOP] > 0.7 * n:
            left_mid_ch = int(0.6 * n)
            left_flag = False

        else:
            left_mid_ch = left_stat[1, cv.CC_STAT_TOP] - 3
            if type == 3:
                left_mid_ch += 1

        if r_height >= 15:
            right_mid_ch = r_down - 13
            right_flag = True

        elif right_stat[1, cv.CC_STAT_TOP] > 2 * n / 3:
            right_mid_ch = int(0.6 * n)
            right_flag = True

        else:
            right_mid_ch = right_stat[1, cv.CC_STAT_TOP] - 3
            if type == 3:
                right_mid_ch += 1

        if False:  # Visualization
            image = uf.gray_to_color(np.array(roi_images)[:, ch_index, :])
            image[left_mid_ch, :mid_pix] = (0, 255, 0)
            image[right_mid_ch, mid_pix:] = (0, 0, 255)
            image[label > 0] = (255, 0, 0)
            plt.imshow(image)
            plt.show()

        if right_flag is True and left_flag is False:
            right_mid_ch = left_mid_ch
        if left_flag is True and right_flag is False:
            left_mid_ch = right_mid_ch

        ch_mid = (int(left_mid_ch), int(right_mid_ch))

    else:
        print("break")
        ch_mid = 4 * ch_image.shape[1] // 10

    n, m, p = np.shape(seg_images)
    pa_region = seg_images[:, pa_index - 10:pa_index + 5, :]  # 20:PA+2
    non_zero_indexes = np.nonzero(pa_region)
    pa_mid = (non_zero_indexes[0][0] + non_zero_indexes[0][-1]) // 2
    dif = abs(r_down - l_down + ch_mid[1] - ch_mid[0]) // 2

    if l_down > r_down:
        pa_mid = (pa_mid + dif, pa_mid)
    elif l_down < r_down:
        pa_mid = (pa_mid, pa_mid + dif)
    else:
        pa_mid = (pa_mid, pa_mid)

    if False:  # Visualization
        image = uf.gray_to_color(np.array(roi_images)[:, pa_index, :])
        image[pa_mid[0], :mid_pix] = (0, 255, 0)
        image[pa_mid[1], mid_pix:] = (0, 0, 255)
        plt.imshow(image)
        plt.show()

    return pa_mid, ch_mid

# Main function for measurement:
# 1. Volume,
# 2. Regional volume: nares, mid-nasal, nasopharynx.
# 3. Surface area.
# 4. Cross-sectional area.
# 5. Trying to calculate length (Not precise enough).
# 6. Inferior data.

def measurement(seg_images, aperture_index, choana_index, thickness, spacing, type):
    """
    Main function for airway measurements including volume, regional volume, surface area, cross-sectional area,
    length, and inferior data.

    :param seg_images: Segmented nasal airway, list of 2D numpy arrays.
    :param aperture_index: Coronal index of the Pyriform Aperture.
    :param choana_index: Coronal index of the Choanae.
    :param thickness: Thickness of each voxel in the image.
    :param spacing: Voxel spacing in the x, y, and z directions.
    :param type: Scan type - 1: normal, 2: head angled up, 3: head angled down.

    :return: List containing dataframes with airway measurements:
             - All airway data: total airway volume, not connected airway volume, regional cross-sectional areas,
               minimal cross-sectional areas, lengths, and surface area.
             - Inferior data: inferior volume and cross-sectional areas for both connected and not connected airways.
    """
    volume_factor = thickness * spacing[0] * spacing[1]

    # Airway lengths
    coronal_images = uf.axial_to_coronal(seg_images)

    # For measuring the length of the airway
    nonzero = np.nonzero(coronal_images)
    first_coronal_pixel = nonzero[0][0]
    end_coronal_pixel = nonzero[0][-1]
    num_of_coronal_slice = (end_coronal_pixel - first_coronal_pixel)
    coronal_len = num_of_coronal_slice * spacing[0]

    # Trying to take into account the influence of the scanning angle and twists
    axial_row_of_first = nonzero[1][0]
    axial_row_of_end = nonzero[1][np.nonzero(nonzero[0] == end_coronal_pixel)[0][0]]
    axial_len = (axial_row_of_end - axial_row_of_first) * thickness
    pitagor_length = math.sqrt(axial_len ** 2 + coronal_len ** 2)

    # Measuring interior airway length
    pa_ch_coronal_len = (choana_index + 1 - aperture_index) * spacing[0]
    pa_ch_length = pitagor_length * pa_ch_coronal_len / coronal_len

    cc_airway = uf.remove_unconnected_objects(coronal_images)  # Connected component of the airway
    not_cc_airway = coronal_images - cc_airway  # The not connected parts

    # Labeling the left or right side of the objects
    cc_label = cc3d.connected_components(cc_airway[:choana_index + 1])  # Labeling of the cc_airway to the 2 airway sides
    not_cc_label = cc3d.connected_components(not_cc_airway[:choana_index + 1])
    cc_label, left_num, right_num = find_side_cc_label_number(cc_label)
    notcc_left_nums, notcc_right_nums = find_side_notcc_label_numbers(not_cc_label)

    # Measuring volume of 1. total airway 2. not cc airway, 3. nostril 4. mid-nasal 5. nasopharynx,
    cc_volume_data, not_cc_volume_data = volume.volume_measurement(
        cc_airway, not_cc_airway, cc_label, not_cc_label, left_num, right_num,
        notcc_left_nums, notcc_right_nums, aperture_index, choana_index, volume_factor)

    # Measuring cross-sectional area
    cc_rcs_data, cc_lcs_data, notcc_rcs_data, notcc_lcs_data = cs.final_cs_measurement(
        cc_airway, not_cc_airway, cc_label, not_cc_label, left_num, right_num, notcc_left_nums, notcc_right_nums,
        aperture_index, choana_index, thickness, spacing, first_coronal_pixel, num_of_coronal_slice)

    # Search the minimal CSA in different regions
    min_cs_area_nares, percent_of_min_nares, min_cs_area_internal, percent_of_min_internal, min_cs_area_pa, \
    percent_of_min_pa = cs.find_min_cs_area(cc_rcs_data, cc_lcs_data, notcc_rcs_data, notcc_lcs_data)

    cc_volume_data["min cs area nares"] = min_cs_area_nares
    cc_volume_data["percent of min nares"] = percent_of_min_nares
    cc_volume_data["min cs area internal"] = min_cs_area_internal
    cc_volume_data["percent of min internal"] = percent_of_min_internal
    cc_volume_data["min cs area pa"] = min_cs_area_pa
    cc_volume_data["percent of min pa"] = percent_of_min_pa

    cc_volume_data["length"] = [round(pitagor_length, 2)]  # Round(coronal_len, 2)
    cc_volume_data["pa-ch length"] = [round(pa_ch_length, 2)]

    # Surface area
    spacing_for_function = (spacing[0], thickness, spacing[1])
    verts, faces = marching_cubes(np.array(coronal_images), level=None, spacing=spacing_for_function,
                                  allow_degenerate=False, method='lewiner')[:2]  # degenerate = False?
    surface_area = mesh_surface_area(verts, faces)
    cc_volume_data["surface area"] = [round(surface_area)]

    all_airway_data = [cc_volume_data, not_cc_volume_data, cc_rcs_data, cc_lcs_data, notcc_rcs_data, notcc_lcs_data]

    # Inferior volume and CSA measurement
    pa_mid, ch_mid = finding_inferior_concha(seg_images, aperture_index, choana_index, type)
    r_inf_cs_data, vol_inf_r, l_inf_cs_data, vol_inf_l, notcc_r_inf_cs_data, notcc_vol_inf_r, notcc_l_inf_cs_data, \
    notcc_vol_inf_l = cs.inferior_cs_measurement(cc_label, not_cc_label, left_num, right_num, notcc_left_nums,
                                                notcc_right_nums, aperture_index, choana_index, pa_mid, ch_mid, thickness,
                                                spacing, first_coronal_pixel, num_of_coronal_slice)

    inferior_cc_vol = pd.DataFrame({"vol r": [vol_inf_r], "vol l": [vol_inf_l]})
    inferior_not_cc_vol = pd.DataFrame({"vol r": [notcc_vol_inf_r], "vol l": [notcc_vol_inf_l]})

    inferior_data = [inferior_cc_vol, inferior_not_cc_vol, r_inf_cs_data, l_inf_cs_data, notcc_r_inf_cs_data,
                     notcc_l_inf_cs_data]

    return [all_airway_data, inferior_data]
