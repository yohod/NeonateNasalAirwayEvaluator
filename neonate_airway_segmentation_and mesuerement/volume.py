import numpy as np
import pandas as pd



def nostril_volume_measurement(cc_nares_label, left_num, right_num, volume_factor):
    """
    Measure the volume of each nostril and check their connection to the mid-nasal airway.

    :param cc_nares_label: Numpy 3D array label of the external nose region objects connected to the internal airway.
    :param left_num: Label number of the left nostril.
    :param right_num: Label number of the right nostril.
    :param volume_factor: Factor to convert voxel count to volume.

    :return: Tuple containing rounded left nostril volume, right nostril volume, left nostril connection status,
             right nostril connection status, additional volume added to the left side, additional volume added to the right side.
    """
    add_to_left_volume = 0
    add_to_right_volume = 0

    left_nostril_volume = np.count_nonzero(cc_nares_label == left_num) * volume_factor
    right_nostril_volume = np.count_nonzero(cc_nares_label == right_num) * volume_factor

    if left_nostril_volume < 25:
        add_to_left_volume = left_nostril_volume
        left_nostril_volume = 0
        left_nare_connection = False
    else:
        left_nare_connection = True

    if right_nostril_volume < 25:
        add_to_right_volume = right_nostril_volume
        right_nostril_volume = 0
        right_nare_connection = False
    else:
        right_nare_connection = True

    return round(left_nostril_volume), round(right_nostril_volume), left_nare_connection, right_nare_connection,\
           round(add_to_left_volume), round(add_to_right_volume)


def midnasal_volume_measurement(label, left_num, right_num, volume_factor):
    """
    Measure the volume of the connected midnasal airway.

    :param label: Label of the midnasal airway.
    :param left_num: Label number of the left side.
    :param right_num: Label number of the right side.
    :param volume_factor: Factor to convert voxel count to volume.

    :return: Tuple containing rounded left airway volume, right airway volume, left choana connection status,
             right choana connection status.
    """
    # Measure volume of the connected midnasal airway
    left_cc_airway_volume = round(np.count_nonzero(label == left_num) * volume_factor)
    right_cc_airway_volume = round(np.count_nonzero(label == right_num) * volume_factor)

    if left_cc_airway_volume < 50:
        left_choana_connection = False
    else:
        left_choana_connection = True

    if right_cc_airway_volume < 50:
        right_choana_connection = False
    else:
        right_choana_connection = True

    return left_cc_airway_volume, right_cc_airway_volume, left_choana_connection, right_choana_connection


def volume_measurement(cc_airway, not_cc_airway, cc_label, not_cc_label, left_num, right_num,
                       notcc_left_nums, notcc_right_nums, aperture_index, choana_index, volume_factor):
    """
    Measure various volumes related to the airway structure.

    :param cc_airway: Connected component of the airway.
    :param not_cc_airway: Not connected parts of the airway.
    :param cc_label: Label of the connected components.
    :param not_cc_label: Label of the not connected components.
    :param left_num: Label number of the left side.
    :param right_num: Label number of the right side.
    :param notcc_left_nums: Label numbers of not connected components on the left side.
    :param notcc_right_nums: Label numbers of not connected components on the right side.
    :param aperture_index: Coronal index of the Pyriform Aperture.
    :param choana_index: Coronal index of the Choanae.
    :param volume_factor: Factor to convert voxel count to volume.

    :return: Two Pandas DataFrames containing airway measurement data - one for connected components (cc) and one for not connected components (not_cc).
    """
    cc_volume_data = {}
    not_cc_volume_data = {}

    # Measuring total volume of airway
    cc_volume_data["cc total volume"] = [round(np.count_nonzero(cc_airway) * volume_factor)]

    # Measuring not connected volume in regions
    not_cc_volume_data["not_cc total volume"] = [round(np.count_nonzero(not_cc_airway) * volume_factor)]

    not_cc_volume_data["not_cc left nostril volume"] = [round(volume_factor *
                    sum([np.count_nonzero(not_cc_label[:aperture_index] == i) for i in notcc_right_nums]))]
    not_cc_volume_data["not_cc right nostril volume"] = [round(volume_factor *
                    sum([np.count_nonzero(not_cc_label[:aperture_index] == i) for i in notcc_left_nums]))]
    not_cc_volume_data["not_cc nostrils volume"] = [not_cc_volume_data["not_cc right nostril volume"][0] + \
                                                   not_cc_volume_data["not_cc left nostril volume"][0]]
    not_cc_volume_data["not_cc left airway volume"] = [round(volume_factor *
                    sum([np.count_nonzero(not_cc_label[aperture_index:] == i) for i in notcc_right_nums]))]
    not_cc_volume_data["not_cc right airway volume"] = [round(volume_factor *
                    sum([np.count_nonzero(not_cc_label[aperture_index:] == i) for i in notcc_left_nums]))]
    not_cc_volume_data["not_cc pa-ch volume"] = [not_cc_volume_data["not_cc right airway volume"][0] +
                                                not_cc_volume_data["not_cc left airway volume"][0]]

    not_cc_volume_data["not_cc nasopharynx volume"] = [round(np.count_nonzero(not_cc_airway[choana_index + 1:]) * volume_factor)]

    # Measuring nostril volume
    left_nostril_volume, right_nostril_volume, left_nare_connection, right_nare_connection, add_left_volume, add_right_volume = \
        nostril_volume_measurement(cc_label[:aperture_index], left_num, right_num, volume_factor)

    cc_volume_data["right nostril cc volume"] = [left_nostril_volume]
    cc_volume_data["left nostril cc volume"] = [right_nostril_volume]
    cc_volume_data["nostrils volume"] = [left_nostril_volume + right_nostril_volume]
    cc_volume_data["right nare connection"] = [left_nare_connection]
    cc_volume_data["left nare connection"] = [right_nare_connection]

    # Measuring midnasal cc airway volume
    left_cc_airway_volume, right_cc_airway_volume, left_choana_connection, right_choana_connection = \
        midnasal_volume_measurement(cc_label[aperture_index:], left_num, right_num, volume_factor)

    cc_volume_data["right cc volume"] = [left_cc_airway_volume + add_left_volume]
    cc_volume_data["left cc volume"] = [right_cc_airway_volume + add_right_volume]
    cc_volume_data["pa-ch volume"] = [cc_volume_data["right cc volume"][0] +
                                      cc_volume_data["left cc volume"][0]]

    # Measuring nasopharynx volume
    nasopharynx = cc_airway[choana_index + 1:]
    nasopharynx_volume = round(np.count_nonzero(nasopharynx) * volume_factor)
    cc_volume_data["nasopharynx volume"] = [nasopharynx_volume]
    if nasopharynx_volume < 50:
        left_choana_connection = False
        right_choana_connection = False

    cc_volume_data["right choana connection"] = [left_choana_connection]
    cc_volume_data["left choana connection"] = [right_choana_connection]

    return pd.DataFrame(cc_volume_data), pd.DataFrame(not_cc_volume_data)
