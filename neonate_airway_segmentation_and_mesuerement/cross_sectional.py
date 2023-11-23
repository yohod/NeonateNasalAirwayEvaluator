import numpy as np
import pandas as pd
import cv2 as cv
from statistics import median
import usefull_function as uf

#
def cs_measuring_slice(image, side_num, height_factor, width_factor, cc_flag=True):
    """
    Measure the cross-sectional area and other cross-sectional data
    of a coronal slice with airway objects from one side.

    :param image: The input coronal slice image.
    :param side_num: The side number to extract airway objects.
    :param height_factor: Factor to scale the height.
    :param width_factor: Factor to scale the width.
    :param cc_flag: Flag to determine if it's a connected nasal airway measure (True) or disconnected (False).
    :return: Tuple containing average height, average width, and maximum width of the measured airway objects.
    """

    help_image = np.copy(image)
    if cc_flag is True:
        help_image[image != side_num] = 0
    num, label, stat, cent = uf.cc_stat(help_image)
    cs_dataset = pd.DataFrame(
        columns=["area", "begin_axial", "end_axial", "height", "avg_width", "min_width", "max_width", "mad_width"])
    sum_width = 0
    sum_len = 0
    avg_height = 0
    for i in range(1, num):
        area = round(stat[i, cv.CC_STAT_AREA] * width_factor * height_factor , 1) # * depth_factor
        top = stat[i, cv.CC_STAT_TOP]
        height = stat[i, cv.CC_STAT_HEIGHT] * height_factor
        down = stat[i, cv.CC_STAT_TOP] + stat[i, cv.CC_STAT_HEIGHT] - 1
        left = stat[i, cv.CC_STAT_LEFT]
        right = stat[i, cv.CC_STAT_LEFT] + stat[i, cv.CC_STAT_WIDTH]

        width_list = []
        for row in range(top, down + 1):
            width = np.count_nonzero(label[row, left:right]) * width_factor
            width_list.append(round(width, 1))
        avg_height += height
        sum_width += sum(width_list)
        sum_len += len(width_list)
        width_avg = round(sum(width_list) / len(width_list), 1)
        width_max = max(width_list)
        width_min = min(width_list)
        width_med = median(width_list)

        obj_data = pd.DataFrame({"area": [area], "begin_axial": [top], "end_axial": [down], "height": [height],
                                 "avg_width": [width_avg], "min_width": [width_min], "max_width": [width_max],
                                 "mad_width": [width_med]})

        cs_dataset = uf.concat(cs_dataset, obj_data)

    if not cs_dataset.empty:
        cs_dataset = cs_dataset.sort_values(by="area", ascending=False)
        avg_width = round(sum_width / sum_len, 1)
        avg_height = round((avg_height / num - 1))
        max_width = cs_dataset["max_width"].max()
        return avg_height, avg_width, max_width  # cs_dataset.loc[0,"avg_width"],
    else:
        return 0, 0, 0



def get_cs_data(cc_label, begin, end, region, thickness, spacing, norm_factors, first_coronal_pixel, left_num=0,
                right_num=0, first_left=0, first_right=0):
    """
    Measure the cross-sectional data of a nasal region (nostril, interior airway, nasopharynx).

    :param cc_label: The label of the nasal airway.
    :param begin: The starting index of the region.
    :param end: The ending index of the region.
    :param region: The region type (1 for nostril, 2 for airway, 3 for nasopharynx).
    :param thickness: The thickness factor of the axial slices.
    :param spacing: The spacing factors in the (coronal, sagittal) directons.
    :param norm_factors: Normalization factors for the distance of the nasal airway.
    :param first_coronal_pixel: The index of the first coronal pixel.
    :param left_num: The label number for the left side.
    :param right_num: The label number for the right side.
    :param first_left: The index of the first coronal pixel for the left side.
    :param first_right: The index of the first coronal pixel for the right side.
    :return: Cross-sectional data for the specified region and sides.
    """
    area_factor = spacing[1]  * thickness #
    cc_r_region_cs_data = pd.DataFrame(
        columns=["region", "side", "coronal index", "x", "percentage of pa-ch", "percentage of nasal airway", "area",
                 "avg_height", "avg_width", "max_width"])
    cc_l_region_cs_data = pd.DataFrame(
        columns=["region", "side", "coronal index", "x", "percentage of pa-ch", "percentage of nasal airway", "area",
                 "avg_height", "avg_width", "max_width"])
    cc_nx_region_cs_data = pd.DataFrame(
        columns=["region", "side", "coronal index", "x", "percentage of pa-ch", "percentage of nasal airway", "area",
                 "avg_height", "avg_width", "max_width"])
    region_dict = {"1": "nares", "2": "midnasal", "3": "nasopharynx"}

    if region < 3:  # nostril or airway. not nasopharynx
        for coronal_index in range(begin, end):
            percent = round(100 * (coronal_index - norm_factors[0]) / norm_factors[1], 1)
            percent_all = round(100 * (coronal_index - norm_factors[2]) / norm_factors[3], 1)
            cc_cs_label = cc_label[coronal_index, :, :]
            if first_left <= coronal_index:
                left_cs_area = area_factor * np.count_nonzero(cc_cs_label == left_num)
                left_cs_data = cs_measuring_slice(cc_cs_label, left_num, thickness, spacing[1])
            else:
                left_cs_area = 0
                left_cs_data = (0, 0, 0)  # need to be changed when include all width cs data

            if first_right <= coronal_index:
                right_cs_area = area_factor * np.count_nonzero(cc_cs_label == right_num)
                right_cs_data = cs_measuring_slice(cc_cs_label, right_num, thickness, spacing[1])
            else:
                right_cs_area = 0
                right_cs_data = (0, 0, 0)

            len_index = round((coronal_index - first_coronal_pixel) * spacing[0], 2)
            dfr = pd.DataFrame({"region": [region_dict[str(region)]], "side": ["right"],
                                "coronal index": [coronal_index], "x": [len_index], "percentage of pa-ch": [percent],
                                "percentage of nasal airway": [percent_all], "area": [round(left_cs_area, 1)],
                                "avg_height": left_cs_data[0], "avg_width": [left_cs_data[1]],
                                "max_width": [left_cs_data[2]]})
            dfl = pd.DataFrame({"region": [region_dict[str(region)]], "side": ["left"],
                                "coronal index": [coronal_index], "x": [len_index], "percentage of pa-ch": [percent],
                                "percentage of nasal airway": [percent_all], "area": [round(right_cs_area, 1)],
                                "avg_height": right_cs_data[0], "avg_width": [right_cs_data[1]],
                                "max_width": [right_cs_data[2]]})
            cc_r_region_cs_data = uf.concat(cc_r_region_cs_data, dfr)
            cc_l_region_cs_data = uf.concat(cc_l_region_cs_data, dfl)
        cc_region_cs_data = [cc_r_region_cs_data, cc_l_region_cs_data]

    else:  # nasopharynx
        for coronal_index in range(begin, end):
            percent = round(100 * (coronal_index - norm_factors[0]) / norm_factors[1], 1)
            percent_all = round(100 * (coronal_index - norm_factors[2]) / norm_factors[3], 1)
            cc_cs_label = cc_label[coronal_index, :, :]
            cs_area = area_factor * np.count_nonzero(cc_cs_label)
            len_index = round((coronal_index - first_coronal_pixel) * spacing[0], 2)
            cs_data = cs_measuring_slice(cc_cs_label, 255, thickness, spacing[1])
            # all the pixel hava the value of 255, so this is the object num
            cc_cs_data = pd.DataFrame({"region": [region_dict[str(region)]], "side": [""],
                                       "coronal index": [coronal_index],
                                       "x": [len_index], "percentage of pa-ch": [percent],
                                       "percentage of nasal airway": [percent_all],
                                       "area": [round(cs_area, 1)], "avg_height": cs_data[0], "avg_width": [cs_data[1]],
                                       "max_width": [cs_data[2]]})
            cc_nx_region_cs_data = uf.concat(cc_nx_region_cs_data, cc_cs_data)
        cc_region_cs_data = [cc_nx_region_cs_data]

    return cc_region_cs_data

def notcc_cs_measurement(not_cc_airway, not_cc_label, notcc_left_nums, notcc_right_nums, aperture_index, choana_index,
                         thickness, spacing, first_coronal_pixel, length):
    """
    Compute cross-sectional area (CSA) measurements of the disconnected components

    :param not_cc_airway: 3D array representing the airway structure excluding connected components.
    :param not_cc_label: 3D array with labeled connected components.
    :param notcc_left_nums: List of labels corresponding to the left side connected components.
    :param notcc_right_nums: List of labels corresponding to the right side connected components.
    :param aperture_index: Index indicating the coronal slice where the nasal aperture begins.
    :param choana_index: Index indicating the coronal slice where the choanae are located.
    :param thickness: Thickness of the axial slices.
    :param spacing: Tuple representing the pixel spacing in the (coronal, sagittal) directions.
    :param first_coronal_pixel: Index of the first coronal pixel in the image.
    :param length: Length of the structure being measured.

    :return: DataFrames containing CSA measurements for the right side, left side, and nasopharynx.
    """

    # Calculate area factor for CSA measurement
    area_factor = spacing[1] * thickness
    # Normalization factors for percentage calculation
    normal_factors = (aperture_index, choana_index - aperture_index, first_coronal_pixel, length)

    # DataFrames to store CSA measurements for left, right, and nasopharynx regions
    notcc_r_cs_data = pd.DataFrame(
        columns=["region", "side", "coronal index", "x", "percentage of pa-ch", "percentage of nasal airway", "area",
                 "avg_height", "avg_width", "max_width"])
    notcc_l_cs_data = pd.DataFrame(
        columns=["region", "side", "coronal index", "x", "percentage of pa-ch", "percentage of nasal airway", "area",
                 "avg_height", "avg_width", "max_width"])
    notcc_nx_cs_data = pd.DataFrame(
        columns=["region", "side", "coronal index", "x", "percentage of pa-ch", "percentage of nasal airway", "area",
                 "avg_height", "avg_width", "max_width"])

    # Dictionary for mapping region codes to region names
    region_dict = {"1": "nares", "2": "midnasal", "3": "nasopharynx"}

    # Create labels for left and right sides
    left_label = not_cc_label.copy()
    for num in notcc_right_nums:
        left_label[left_label == num] = 0

    right_label = not_cc_label.copy()
    for num in notcc_left_nums:
        right_label[right_label == num] = 0

    # Extract indices where labels are non-zero
    left_where = one_time_item(list(np.where(left_label != 0)[0]))
    right_where = one_time_item(list(np.where(right_label != 0)[0]))

    # Extract indices where nasopharynx label is non-zero
    naso_label = not_cc_airway[choana_index + 1:, :, :]
    n, _, _ = naso_label.shape
    naso_where = one_time_item(list(np.where(naso_label != 0)[0]))

    # Iterate through coronal slices for CSA measurement
    for coronal_index in range(first_coronal_pixel, choana_index + 1):
        # Calculate percentage values
        percent = round(100 * (coronal_index - normal_factors[0]) / normal_factors[1], 1)
        percent_all = round(100 * (coronal_index - normal_factors[2]) / normal_factors[3], 1)
        len_index = round((coronal_index - first_coronal_pixel) * spacing[0], 2)

        # CSA measurement for the left side
        if len(left_where) == 0 or coronal_index != left_where[0]:
            left_cs_area = 0
            left_cs_data = (0, 0, 0)
        else:
            del left_where[0]
            image = left_label[coronal_index, :, :]
            left_cs_area = np.count_nonzero(image) * area_factor
            left_cs_data = cs_measuring_slice(image, 1, thickness, spacing[1], cc_flag=False)

        # CSA measurement for the right side
        if len(right_where) == 0 or coronal_index != right_where[0]:
            right_cs_area = 0
            right_cs_data = (0, 0, 0)
        else:
            del right_where[0]
            image = right_label[coronal_index, :, :]
            right_cs_area = np.count_nonzero(image) * area_factor
            right_cs_data = cs_measuring_slice(image, 1, thickness, spacing[1], cc_flag=False)

        # Determine the region based on coronal index
        if coronal_index < aperture_index:
            region = 1  # nostril region
        else:
            region = 2  # mid-nasal region

        # Create DataFrames for CSA data for the left and right sides
        dfr = pd.DataFrame({"region": [region_dict[str(region)]], "side": ["right"],
                            "coronal index": [coronal_index], "x": [len_index], "percentage of pa-ch": [percent],
                            "percentage of nasal airway": [percent_all], "area": [round(left_cs_area, 1)],
                            "avg_height": left_cs_data[0], "avg_width": [left_cs_data[1]],
                            "max_width": [left_cs_data[2]]})
        dfl = pd.DataFrame({"region": [region_dict[str(region)]], "side": ["left"],
                            "coronal index": [coronal_index], "x": [len_index], "percentage of pa-ch": [percent],
                            "percentage of nasal airway": [percent_all], "area": [round(right_cs_area, 1)],
                            "avg_height": right_cs_data[0], "avg_width": [right_cs_data[1]],
                            "max_width": [right_cs_data[2]]})
        notcc_r_cs_data = uf.concat(notcc_r_cs_data, dfr)
        notcc_l_cs_data = uf.concat(notcc_l_cs_data, dfl)

    # Iterate through remaining coronal slices for nasopharynx CSA measurement
    end = normal_factors[3] + first_coronal_pixel - choana_index
    for coronal_index in range(end):
        percent = round(100 * (coronal_index + choana_index + 1 - normal_factors[0]) / normal_factors[1], 1)
        percent_all = round(100 * (coronal_index + choana_index + 1 - normal_factors[2]) / normal_factors[3], 1)
        len_index = round((coronal_index + choana_index + 1 - first_coronal_pixel) * spacing[0], 2)

        # CSA measurement for the nasopharynx
        if len(naso_where) == 0 or coronal_index != naso_where[0]:
            cs_area = 0
            cs_data = (0, 0, 0)
        else:
            del naso_where[0]
            image = naso_label[coronal_index, :, :]
            cs_area = np.count_nonzero(image) * area_factor
            cs_data = cs_measuring_slice(image, 1, thickness, spacing[1], cc_flag=False)

        # Create DataFrame for nasopharynx CSA data
        region = 3  # Nasopharynx region
        cc_cs_data = pd.DataFrame(
            {"region": [region_dict[str(region)]], "side": [""], "coronal index": [coronal_index + choana_index + 1],
             "x": [len_index], "percentage of pa-ch": [percent], "percentage of nasal airway": [percent_all],
             "area": [round(cs_area, 1)], "avg_height": cs_data[0],
             "avg_width": [cs_data[1]], "max_width": [cs_data[2]]})
        notcc_nx_cs_data = uf.concat(notcc_nx_cs_data, cc_cs_data)
    notcc_r_cs_data = uf.concat(notcc_r_cs_data, notcc_nx_cs_data)
    notcc_l_cs_data = uf.concat(notcc_l_cs_data, notcc_nx_cs_data)

    return notcc_r_cs_data, notcc_l_cs_data

def one_time_item(list):
    ret_list = []
    last = -1
    for item in list:
        if item > last:
            last = item
            ret_list.append(item)
    return ret_list

# main function
def final_cs_measurement(cc_airway, not_cc_airway, cc_label, not_cc_label, left_num, right_num,
                         notcc_left_nums, notcc_right_nums, aperture_index, choana_index, thickness, spacing,
                         first_coronal_pixel, length):
    """
    Measure the cross-sectional area of the entire nasal airway, divided into left and right sides,
    and three different regions (external, internal, nasopharynx).

    :param cc_airway: 3D array representing the connected components of the nasal airway.
    :param not_cc_airway: 3D array representing the non-connected components of the nasal airway.
    :param cc_label: 3D array with labeled connected components.
    :param not_cc_label: 3D array with labeled non-connected components.
    :param left_num: Label number corresponding to the left side connected component.
    :param right_num: Label number corresponding to the right side connected component.
    :param notcc_left_nums: List of labels corresponding to the left side non-connected components.
    :param notcc_right_nums: List of labels corresponding to the right side non-connected components.
    :param aperture_index: Index indicating the coronal slice where the nasal aperture begins.
    :param choana_index: Index indicating the coronal slice where the choanae are located.
    :param thickness: Thickness of the axial slices.
    :param spacing: Tuple representing the pixel spacing in the (coronal, sagittal) directions.
    :param first_coronal_pixel: Index of the first coronal pixel in the image.
    :param length: Length of the structure being measured.

    :return: DataFrames containing cross-sectional area measurements for the right side, left side, and nasopharynx
             for both connected and non-connected components.
    """
    normal_factors = (aperture_index, choana_index - aperture_index, first_coronal_pixel, length)

    # Find the first pixel of the left and right sides in the connected components
    if left_num > 0:
        first_left = np.where(cc_label == left_num)[0][0]
    else:
        first_left = choana_index + 1
    if right_num > 0:
        first_right = np.where(cc_label == right_num)[0][0]
    else:
        first_right = choana_index + 1

    # Nostril, Region = 1
    cc_nostril_cs_data = get_cs_data(cc_label, first_coronal_pixel,
                                     aperture_index, 1, thickness, spacing, normal_factors, first_coronal_pixel,
                                     left_num, right_num, first_left, first_right)
    cc_rnostril_cs_data = cc_nostril_cs_data[0]
    cc_lnostril_cs_data = cc_nostril_cs_data[1]

    # Airway, Region = 2
    cc_pa_ch_cs_data = get_cs_data(cc_label, aperture_index, choana_index + 1, 2,
                                   thickness, spacing, normal_factors, first_coronal_pixel, left_num, right_num,
                                   first_left, first_right)
    cc_rpa_ch_cs_data = cc_pa_ch_cs_data[0]
    cc_lpa_ch_cs_data = cc_pa_ch_cs_data[1]

    # Nasopharynx, Region = 3
    end = normal_factors[3] + first_coronal_pixel + 1
    cc_nasopharynx_cs_data = get_cs_data(cc_airway, choana_index + 1, end, 3, thickness, spacing, normal_factors,
                                         first_coronal_pixel)[0]

    # Concatenate data for right side
    r_cs = pd.concat([cc_rnostril_cs_data, cc_rpa_ch_cs_data, cc_nasopharynx_cs_data], ignore_index=True)
    r_cs = r_cs.drop(labels="side", axis='columns')

    # Concatenate data for left side
    l_cs = pd.concat([cc_lnostril_cs_data, cc_lpa_ch_cs_data, cc_nasopharynx_cs_data], ignore_index=True)
    l_cs = l_cs.drop(labels="side", axis='columns')

    # Measure cross-sectional area for non-connected components
    notcc_r_sc, notcc_l_sc = notcc_cs_measurement(not_cc_airway, not_cc_label, notcc_left_nums, notcc_right_nums,
                                                  aperture_index,
                                                  choana_index, thickness, spacing, first_coronal_pixel, length)

    return r_cs, l_cs, notcc_r_sc, notcc_l_sc



# CSA data of the inferior part of the nasal airway
def inferior_cs_measurement(cc_label, not_cc_label, left_num, right_num, notcc_left_nums, notcc_right_nums,
                           aperture_index, choana_index, pa_mid, ch_mid, thickness, spacing, first_coronal_pixel,
                           length):
    norm_factor = (aperture_index, choana_index - aperture_index, first_coronal_pixel, length)
    area_factor = thickness * spacing[1]
    slopel = (ch_mid[0] - pa_mid[0]) / (choana_index - aperture_index)
    sloper = (ch_mid[1] - pa_mid[1]) / (choana_index - aperture_index)
    left_intercept = ch_mid[0] - slopel * choana_index
    right_intercept = ch_mid[1] - sloper * choana_index
    cc_r_cs_data = pd.DataFrame(
        columns=["region", "side", "coronal index", "x", "percentage of pa-ch", "percentage of nasal airway", "area",
                 "avg_height", "avg_width", "max_width"])
    cc_l_cs_data = pd.DataFrame(
        columns=["region", "side", "coronal index", "x", "percentage of pa-ch", "percentage of nasal airway", "area",
                 "avg_height", "avg_width", "max_width"])
    not_cc_r_cs_data = pd.DataFrame(
        columns=["region", "side", "coronal index", "x", "percentage of pa-ch", "percentage of nasal airway", "area",
                 "avg_height", "avg_width", "max_width"])
    not_cc_l_cs_data = pd.DataFrame(
        columns=["region", "side", "coronal index", "x", "percentage of pa-ch", "percentage of nasal airway", "area",
                 "avg_height", "avg_width", "max_width"])

    for coronal_index in range(aperture_index, choana_index + 1):
        percent = round(100 * (coronal_index - norm_factor[0]) / norm_factor[1], 1)
        percent_all = round(100 * (coronal_index - norm_factor[2]) / norm_factor[3], 1)
        len_index = round((coronal_index - first_coronal_pixel) * spacing[0], 2)

        # mesuring the cc infirior data
        cc_cs_label = cc_label[coronal_index, :, :]

        # left side information
        left_mid = int(round(slopel * coronal_index + left_intercept))
        image = cc_cs_label[left_mid:, :]
        left_cs_area = area_factor * np.count_nonzero(image == left_num)
        left_cs_data = cs_measuring_slice(image, left_num, thickness, spacing[1])

        # right side information
        right_mid = int(round(sloper * coronal_index + right_intercept))
        image = cc_cs_label[right_mid:, :]
        right_cs_area = area_factor * np.count_nonzero(image == right_num)
        right_cs_data = cs_measuring_slice(image, right_num, thickness, spacing[1])

        # add the slice information to the dataframe
        dfr = pd.DataFrame({"region": ["airway"], "side": ["right"],
                            "coronal index": [coronal_index], "x": [len_index], "percentage of pa-ch": [percent],
                            "percentage of nasal airway": [percent_all], "area": [round(left_cs_area, 1)],
                            "avg_height": left_cs_data[0], "avg_width": [left_cs_data[1]],
                            "max_width": [left_cs_data[2]]})
        dfl = pd.DataFrame({"region": ["airway"], "side": ["left"],
                            "coronal index": [coronal_index], "x": [len_index], "percentage of pa-ch": [percent],
                            "percentage of nasal airway": [percent_all], "area": [round(right_cs_area, 1)],
                            "avg_height": right_cs_data[0], "avg_width": [right_cs_data[1]],
                            "max_width": [right_cs_data[2]]})
        cc_r_cs_data = uf.concat(cc_r_cs_data, dfr)
        cc_l_cs_data = uf.concat(cc_l_cs_data, dfl)

        # mesuring the dis connected airway infirior data.
        not_cc_cs_label = not_cc_label[coronal_index, :, :]

        # left side
        image = not_cc_cs_label[left_mid:, :].copy()
        for num in notcc_right_nums:  # erase all the object from the right side
            image[image == num] = 0
        left_cs_area = area_factor * np.count_nonzero(image)
        if left_cs_area > 0:
            left_cs_data = cs_measuring_slice(image, notcc_left_nums, thickness, spacing[1], cc_flag=False)
        else:
            left_cs_data = 0, 0, 0
        # right side
        image = not_cc_cs_label[right_mid:, :].copy()
        for num in notcc_left_nums:  # erase all the object from the left side
            image[image == num] = 0
        right_cs_area = area_factor * np.count_nonzero(image)
        if right_cs_area > 0:
            right_cs_data = cs_measuring_slice(image, notcc_right_nums, thickness, spacing[1], cc_flag=False)
        else:
            right_cs_data = 0, 0, 0

        dfr = pd.DataFrame({"region": ["airway"], "side": ["right"], "coronal index": [coronal_index],
                            "x": [len_index], "percentage of pa-ch": [percent],
                            "percentage of nasal airway": [percent_all],
                            "area": [round(left_cs_area, 1)],
                            "avg_height": left_cs_data[0], "avg_width": [left_cs_data[1]],
                            "max_width": [left_cs_data[2]]})
        dfl = pd.DataFrame({"region": ["airway"], "side": ["left"], "coronal index": [coronal_index],
                            "x": [len_index], "percentage of pa-ch": [percent],
                            "percentage of nasal airway": [percent_all],
                            "area": [round(right_cs_area, 1)],
                            "avg_height": right_cs_data[0], "avg_width": [right_cs_data[1]],
                            "max_width": [right_cs_data[2]]})

        not_cc_r_cs_data = uf.concat(not_cc_r_cs_data, dfr)
        not_cc_l_cs_data = uf.concat(not_cc_l_cs_data, dfl)

<<<<<<< Updated upstream
    vol_r = round(cc_r_cs_data['area'].sum())
    vol_l = round(cc_l_cs_data['area'].sum())
    not_cc_vol_r = round(not_cc_r_cs_data['area'].sum())
    not_cc_vol_l = round(not_cc_l_cs_data['area'].sum())
=======
    vol_r = round(cc_r_cs_data['area'].sum() * spacing[0])
    vol_l = round(cc_l_cs_data['area'].sum() * spacing[0])
    not_cc_vol_r = round(not_cc_r_cs_data['area'].sum() * spacing[0])
    not_cc_vol_l = round(not_cc_l_cs_data['area'].sum() * spacing[0])
>>>>>>> Stashed changes

    return cc_r_cs_data, vol_r, cc_l_cs_data, vol_l, not_cc_r_cs_data, not_cc_vol_r, not_cc_l_cs_data, not_cc_vol_l


def find_min_cs_area(cc_rcs_data, cc_lcs_data, notcc_rcs_data, notcc_lcs_data):
    """
    Find the minimum cross-sectional area and extract specific information from the corresponding row.

    :param cc_rcs_data: DataFrame containing cross-sectional area data for cc_rcs
    :param cc_lcs_data: DataFrame containing cross-sectional area data for cc_lcs
    :param notcc_rcs_data: DataFrame containing cross-sectional area data for notcc_rcs
    :param notcc_lcs_data: DataFrame containing cross-sectional area data for notcc_lcs
    :return: Tuple containing the minimum cross-sectional area, corresponding percentage of nasal airway,
             area, and x-coordinate
    """
    # Create a copy of cc_rcs_data to avoid modifying the original DataFrame
    df = cc_rcs_data.copy()
    df["area"] = pd.to_numeric(df["area"])
    # Calculate the total area by adding the area values from all DataFrames
    df["area"] = cc_rcs_data["area"] + cc_lcs_data["area"] + notcc_rcs_data["area"] + notcc_lcs_data["area"]

    # Filter the DataFrame based on the percentage of nasal airway criteria
    df = df[(df["percentage of nasal airway"] > 5) & (df["percentage of nasal airway"] < 80)]

    # Find the row with the minimum area and extract specific information
    df = df.sort_values(["area", "percentage of nasal airway"], ascending=[True, True])  # Sort the DataFrame based on the "area" column
    min_row_information = df[df["percentage of pa-ch"] < 0].iloc[0]  # Retrieve the first row with the minimum area
    min_cs_area_nares = min_row_information["area"]
    percent_of_min_nares = min_row_information["percentage of nasal airway"]

    min_row_information = df[df["percentage of pa-ch"] >= 0].iloc[0]  # Retrieve the first row with the minimum area
    min_cs_area_internal = min_row_information["area"]
    percent_of_min_internal = min_row_information["percentage of nasal airway"]

    min_row_information = df[(df["percentage of pa-ch"] >= 0) & (df["percentage of pa-ch"] < 15)].iloc[0]
    min_cs_area_pa = min_row_information["area"]
    percent_of_min_pa = min_row_information["percentage of nasal airway"]
    return min_cs_area_nares, percent_of_min_nares, min_cs_area_internal, percent_of_min_internal, \
        min_cs_area_pa, percent_of_min_pa
