import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox


import preprocessing as pre
import roi
import usefull_function as uf
import segmentation as segment
import presentation
import measurement as measure
import model_3d as m3d
import measurement_visulization as visual
import standardization as standard
import bone_distance as bone
import validation
import statistical_test as stest


# main function for the Segmentation, save images, measurement, plot them and save them, 3d model, bone distance
def main_measurement(path=None, save_path=None, plot_measure=True, model3d=True, saving_axial=True,
                     axial_anim=False, saving_coronal=True, coronal_anim=False, save_data=True,
                     pa_width=True):
    """
    Perform segmentation, measurements, and visualization on medical images.

    :param path: Path to the folder of the neonate CT images(dicom files).
    :param save_path: Path to save the results.
    :param plot_measure: Whether to plot measurement data.
    :param model3d: Whether to generate a 3D model.
    :param saving_axial: Whether to save the segmentation result in axial images.
    :param axial_anim: Whether to create an animation for axial images.
    :param saving_coronal: Whether to save the segmentation result in coronal images.
    :param coronal_anim: Whether to create an animation for coronal images.
    :param save_data: Whether to save measurement data to an Excel file.
    :param pa_width: Whether to measure Pyriform aperture (PA) width .

    :return: None
    """
    # Handle default values for path and save_path
    if path is None:
        path = uf.select_location("Select Case Location")
    if save_path is None:
        save_path = uf.select_location("Select Save Location")

    # Check if the save_path directory exists, create it if not
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    os.chdir(save_path)

    # Perform preprocessing on the medical images
    slices, images_for_sagittal, images, spacing, reverse = pre.preprocessing(path)

    # Perform region of interest (ROI) selection on the images
    slices, all_head, images_for_sagittal, images, edge_index, end_open_nose_index, naso_index, \
        end_nasopharynx, image_type, interference = roi.voi(slices, images_for_sagittal, images)

    # Perform nasal airway segmentation on the images
    slices, all_head, roi_images, seg_images, fix_seg_images, open_nostril_index, edge_index = \
        segment.seg_nasal_airway(slices, all_head, images, edge_index, end_open_nose_index, naso_index,
                                 end_nasopharynx, image_type)

    # (thickness- axial spacing, spacing[1]- coronal spacing, spacing[2] - sagittal spacing)
    thickness = spacing[0]
    spacing = (float(spacing[1]), float(spacing[2]))

    # find the PA and choana indices
    aperture_index = measure.find_aperture(roi_images, edge_index, image_type)
    choana_index = measure.find_choana(fix_seg_images, roi_images)

    # Save axial images if enabled
    if saving_axial:
        axial_images = presentation.presentation(slices, roi_images, seg_images, fix_seg_images,
                                                 plane_mode="axial", additional_slices=all_head)
        presentation.save_img(axial_images, save_path=save_path, mode='axial', anim=axial_anim)

    # Save coronal images if enabled
    if saving_coronal:
        coronal_images = presentation.presentation(slices, roi_images, seg_images, fix_seg_images,
                                                   plane_mode="coronal")
        presentation.save_img(coronal_images, save_path=save_path, mode='coronal', anim=coronal_anim)

    # Generate 3D model if enabled
    if model3d:
        m3d.reconstruction3d(fix_seg_images[:, :, :], spacing, thickness, save_path=save_path, connected=True)

    # Measure various parameters
    measurement_data = measure.measurement(fix_seg_images, aperture_index, choana_index, thickness, spacing, image_type)

    # Plot measurement data if enabled
    if plot_measure:
        visual.plot(measurement_data[0], aperture_index, choana_index, save_path=save_path)
        # visual.plot_inferior(measurement_data[1], aperture_index, choana_index, save_path=save_path)

    # Save measurement data to Excel file if enabled
    if save_data:
        excel_path = os.path.join(save_path, 'measurements.xlsx')
        sheet_name = [
            ['cc volume', 'not cc volume', 'cs r data', 'cs l data', 'notcc cs r data', 'notcc cs l data'],
            ['inferior cc volume', 'inferior not cc volume', 'inferior cs r data', 'inferior cs l data',
             'inferior notcc cs r data', 'inferior notcc cs l data']]
        with pd.ExcelWriter(excel_path) as writer:
            for i in range(0, 12):
                mode_index = i // 6
                data_index = i % 6
                measurement_data[mode_index][data_index].to_excel(writer, sheet_name=sheet_name[mode_index][data_index])

    # Measure PA width if enabled
    if pa_width:

        bone.measure_bone_distance(fix_seg_images, roi_images, aperture_index, choana_index, thickness=thickness,
                                   spacing=spacing, save_path=save_path)


# analysing functions for "evaluating CNPAS" study

# Function for standardization and visualization of CSA measurement data for three groups:
# 1. normal, 2. moderate (CNPAS cases treated without surgery), 3. severe (CNPAS cases treated by surgery)
def main_standardization_3_all(cases=range(0, 29), exclude_cases=(18, 19, 28), saving_data=False, path=None):
    """
    Perform standardization and visualization of CSA measurement data for three groups: normal, obstruct, and surgery.
    :param cases: List of case indices.
    :param saving_data: Whether to save the standardized data to an Excel file.
    :param path: Path to the data location.
    :param exclude_cases: List of case indices to exclude.
    :return:None
    """

    # Load all Excel data frames
    if path is None:
        path = uf.select_location("Select data Location")

    normal_list = []
    obstruct_list = []  # moderate CNPAS
    surgery_list = []  # severe CNPAS
    pa_percent_list = []
    ch_percent_list = []

    for mode in ["all", "2connected", "1"]:
        # "all" CSA - of two nasal sides. "2connected"- only the CSA of the connected airway. "1"- the average CSA of the two sides
        if mode == "1" or mode == "2connected":
            continue
        for i in cases:
            print(i)
            if i in [1, 2, 9, 24] or i in exclude_cases:  # Skipping cases
                continue
            exel_path = path + "\\" + uf.CASE_MAP[str(i)] + "(" + str(i) + ")" + '.xlsx'

            # Read CSA measurement data and normalize it
            df, pa_percent, ch_percent = standard.read(exel_path, mode=mode, inferior=False,
                                                       percent_method="percentage of nasal airway")
            pa_percent_list.append(pa_percent)
            ch_percent_list.append(ch_percent)

            # Divides the cases into their groups
            if i in [0, 4, 6, 7, 10, 11]:
                surgery_list += df
            elif i in [3, 5, 8, 12]:
                obstruct_list += df
            else:
                normal_list += df

    all_lists = [normal_list, obstruct_list, surgery_list, ]

    stand_df = []
    for df_list in all_lists:
        # Standardize cases, Calculate the average CSA and STD along the normalized nasal cavity for each group.
        avg_df, std_df = standard.standardize_cases(df_list, percent_col_name="percentage of nasal airway")
        stand_df.append([avg_df, std_df])

    if saving_data:
        # Save standardized data to Excel
        excel_path = path + "\\" + "standardization" + '(' + mode + ')' + '.xlsx'
        with pd.ExcelWriter(excel_path) as writer:
            stand_df[0][0].to_excel(writer, sheet_name='averaging normal')
            stand_df[0][1].to_excel(writer, sheet_name='std normal')
            stand_df[1][0].to_excel(writer, sheet_name='averaging obstruct')
            stand_df[1][1].to_excel(writer, sheet_name='std obstruct')
            stand_df[2][0].to_excel(writer, sheet_name='averaging surgery')
            stand_df[2][1].to_excel(writer, sheet_name='std surgery')

    if len(normal_list) == 1:
        std_flag = False
    else:
        std_flag = True

    # calculating the average PA and Choana
    pa_percent = np.round(np.mean(pa_percent_list), 1)
    ch_percent = np.round(np.mean(ch_percent_list), 1)
    # produce a qualitative graph
    standard.plot_compare_3(stand_df, std_flag=std_flag, mode=mode, percent_mode="all",
                            pa_percent=pa_percent, ch_percent=ch_percent, units="mm", save_path=path)


def main_region_averaging(cases=range(0, 29), path=None, exclude_cases=(18, 19, 28), saving_data=True, ttest=True):
    """
    Calculate the average cross-sectional area (CSA) in specific regions (PA, 25%, 50%, 75%, 100%) and perform statistical analysis.

    :param cases: List of case indices.
    :param path: Path to the measurement dictionary.
    :param exclude_cases: List of case indices to exclude.
    :param saving_data: Flag to indicate whether to save the data to Excel.
    :param ttest: Flag to perform t-test in addition to Mann-Whitney U test.
    :return: None
    """
    # Load all Excel data frames
    if path is None:
        path = uf.select_location("Select the measurement dictionary")

    # DataFrame to store CSA data for all cases
    df_all = pd.DataFrame(columns=['case', 'region', 'vol', 'avg area', 'std', 'diagnose'])

    for i in cases:
        print(i)
        if i in [1, 2, 9, 24] or i in exclude_cases:
            continue

        # define the Excel file path
        exel_path = path + "\\" + uf.CASE_MAP[str(i)] + "(" + str(i) + ")" + '.xlsx'

        # Define the diagnosis category based on case index
        if i in [0, 4, 6, 7, 10, 11]:
            diagnose = 'CNPAS + surgery'
        elif i in [3, 5, 8, 12]:
            diagnose = 'CNPAS WO surgery'
        else:
            diagnose = 'normal'

        # Loop through regions and calculate CSA data
        for region in ["PA", "25%", "50%", "75%", "CH"]:
            # Define the percentage range for each region
            if region == "PA":
                min_percent = -5
                max_percent = 5
            elif region == "CH":
                min_percent = 95
                max_percent = 105
            else:
                digit_percent = int(region[:2])
                min_percent = digit_percent - 5
                max_percent = digit_percent + 5

            # Calculate volume, average area, and standard deviation
            vol, avg, std = standard.average(exel_path, min_percent=min_percent, max_percent=max_percent)

            # Create a DataFrame for the case data
            case_data = pd.DataFrame(
                {'case': [i], 'region': [region], 'vol': [vol], 'avg area': [avg], 'std': [std], 'diagnose': [diagnose]})

            # Concatenate case data to the overall DataFrame
            df_all = pd.concat([df_all, case_data], ignore_index=True)

    # Statistical analysis
    statistical_df = pd.DataFrame(columns=['region', 'test', 'avg(1)', 'std(1)', 'p1(mwt)', 'avg(2)', 'std(2)',
                                  'p2(mwt)', 'avg(3)', 'std(3)', 'p3(mwt)', 'avg(4)', 'std(4)', 'p4(mwt)'])

    for region in ["PA", "25%", "50%", "75%", "CH"]:
        # Take only the CSA data of the region
        df_region = df_all[df_all['region'] == region]

        # Divide the CSA information according to the 3 groups
        surgery_data = df_region[(df_region['diagnose'] == 'CNPAS + surgery')]['avg area']
        obstruct_data = df_region[(df_region['diagnose'] == 'CNPAS WO surgery')]['avg area']
        normal_data = df_region[(df_region['diagnose'] == 'normal')]['avg area']

        # Perform statistical tests
        statistical_df = pd.concat([statistical_df, stest.p_val(df_norm=normal_data, df_obstruct=obstruct_data,
                                    df_surgery=surgery_data, title=region, test='mwu')])
        if ttest:
            statistical_df = pd.concat([statistical_df,
                                       stest.p_val(df_norm=normal_data, df_obstruct=obstruct_data,
                                        df_surgery=surgery_data, title=region, test='ttest')], ignore_index=True)

    if saving_data:
        # Save the data to Excel for each region
        excel_path = path + "\\" + "averaging_region_area.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            for region in ["PA", "25%", "50%", "75%", "CH"]:
                df_all[df_all['region'] == region].to_excel(writer, sheet_name=region)
                statistical_df.to_excel(writer, sheet_name="statistical_results")
    else:
        print(statistical_df)


def main_general_data(cases=range(0, 29), path=None, exclude_cases=(18, 19, 28), saving_data=True, ttest=True):
    """
     Calculate general data including volumes, surface area, and perform statistical tests for specified regions.

    :param cases: List of case indices.
    :param path: Path to the measurement dictionary.
    :param exclude_cases: List of case indices to exclude.
    :param saving_data: Flag to indicate whether to save the data to Excel.
    :param ttest: Flag to perform t-test in addition to Mann-Whitney U test.
    :return: None
    """
    # Load all Excel data frames
    if path is None:
        path = uf.select_location("Select the measurement dictionary")

    # DataFrame to store general data for all cases
    df_all = pd.DataFrame(
        columns=['case',
                 'Total volume',
                 'vol',
                 'not c vol',
                 'surface area',
                 'nares volume',
                 'vol nostrils',
                 'not c nostrils',
                 'midnasal volume',
                 'vol midnasal',
                 'not c midnasal',
                 'nasopharynx volume',
                 'vol naso',
                 'not c naso',
                 'inferior volume',
                 'inferior vol',
                 'inferior not cc vol',
                 'min cs area nares',
                 'percent of min nares',
                 'min cs area internal',
                 'percent of min internal',
                 'min cs area pa',
                 'percent of min pa',
                 'len all',
                 'len pa-ch', 'diagnose']
    )

    for i in cases:
        print(i)
        if i in [1, 2, 9, 24] or i in exclude_cases:
            continue

        exel_path = path + "\\" + uf.CASE_MAP[str(i)] + "(" + str(i) + ")" + '.xlsx'

        # Define the diagnosis category based on case index
        if i in [0, 4, 6, 7, 10, 11]:
            diagnose = 'CNPAS + surgery'
        elif i in [3, 5, 8, 12]:
            diagnose = 'CNPAS WO surgery'
        else:
            diagnose = 'normal'

        # Read volumes from Excel sheets
        vol_df = pd.read_excel(exel_path, sheet_name="cc volume")
        not_c_vol_df = pd.read_excel(exel_path, sheet_name="not cc volume")
        inferior_df = pd.read_excel(exel_path, sheet_name="inferior cc volume")
        not_cc_inferior_df = pd.read_excel(exel_path, sheet_name="inferior not cc volume")

        # Create a DataFrame for case data
        all_total_volume = vol_df.iloc[0]["cc total volume"] + not_c_vol_df.iloc[0]["not_cc total volume"]
        all_nares_volume = vol_df.iloc[0]["nostrils volume"] + not_c_vol_df.iloc[0]["not_cc nostrils volume"]
        all_midnasal_volume = vol_df.iloc[0]["pa-ch volume"] + not_c_vol_df.iloc[0]["not_cc pa-ch volume"]
        all_naso_volume = not_c_vol_df.iloc[0]["not_cc nasopharynx volume"] + vol_df.iloc[0]["nasopharynx volume"]
        all_inferior_volume = inferior_df.iloc[0]["vol r"] + inferior_df.iloc[0]["vol l"] + \
            not_cc_inferior_df.iloc[0]["vol r"] + not_cc_inferior_df.iloc[0]["vol l"]

        case_data = pd.DataFrame({'case': [i], 'Total volume': all_total_volume, 'vol': vol_df.iloc[0]["cc total volume"],
                                  'not c vol': not_c_vol_df.iloc[0]["not_cc total volume"],
                                  'surface area': vol_df.iloc[0]["surface area"],
                                  'nares volume': all_nares_volume,
                                  'vol nostrils': vol_df.iloc[0]["nostrils volume"],
                                  'not c nostrils': not_c_vol_df.iloc[0]["not_cc nostrils volume"],
                                  'midnasal volume': all_midnasal_volume,
                                  'vol midnasal': vol_df.iloc[0]["pa-ch volume"],
                                  'not c midnasal': not_c_vol_df.iloc[0]["not_cc pa-ch volume"],
                                  'nasopharynx volume': all_naso_volume,
                                  'vol naso': vol_df.iloc[0]["nasopharynx volume"],
                                  'not c naso': not_c_vol_df.iloc[0]["not_cc nasopharynx volume"],
                                  'inferior volume': all_inferior_volume,
                                  'inferior vol': inferior_df.iloc[0]["vol r"] + inferior_df.iloc[0]["vol l"],
                                  'inferior not cc vol': not_cc_inferior_df.iloc[0]["vol r"] + not_cc_inferior_df.iloc[0]["vol l"],
                                  'min cs area nares': vol_df.iloc[0]["min cs area nares"],
                                  'percent of min nares': vol_df.iloc[0]["percent of min nares"],
                                  'min cs area internal': vol_df.iloc[0]["min cs area internal"],
                                  'percent of min internal': vol_df.iloc[0]["percent of min internal"],
                                  'min cs area pa': vol_df.iloc[0]["min cs area pa"],
                                  'percent of min pa': vol_df.iloc[0]["percent of min pa"],
                                  'len all': vol_df.iloc[0]["length"],
                                  'len pa-ch': vol_df.iloc[0]["pa-ch length"], 'diagnose': [diagnose]})

        df_all = uf.concat(df_all, case_data)

    # Statistical analysis
    statistical_df = pd.DataFrame(columns=['region', 'test', 'avg(1)', 'std(1)', 'p1(mwt)', 'avg(2)', 'std(2)',
                                           'p2(mwt)', 'avg(3)', 'std(3)', 'p3(mwt)', 'avg(4)', 'std(4)', 'p4(mwt)'])

    # List of regions and corresponding volume columns
    regions_and_volumes = ['Total volume', 'nares volume', 'midnasal volume',
                           'nasopharynx volume', 'surface area']

    for region in regions_and_volumes:
        if region == "nasopharynx volume":
            alternative = "greater"
        else:
            alternative = "less"

        # Divides the region information according to the 3 groups
        surgery_data = df_all[df_all['diagnose'] == 'CNPAS + surgery'][region].astype(int)
        obstruct_data = df_all[df_all['diagnose'] == 'CNPAS WO surgery'][region].astype(int)
        normal_data = df_all[df_all['diagnose'] == 'normal'][region].astype(int)

        # Perform statistical tests
        statistical_df = uf.concat(statistical_df, stest.p_val(df_norm=normal_data, df_obstruct=obstruct_data,
                                   df_surgery=surgery_data, title=region, test='mwu', alternative=alternative))
        if ttest is True:
            statistical_df = pd.concat([statistical_df,
                                        stest.p_val(df_norm=normal_data, df_obstruct=obstruct_data, df_surgery=surgery_data, title=region,
                                                    test='ttest', alternative=alternative)], ignore_index=True)

    if saving_data:
        # Save the data to Excel
        excel_path = path + "\\" + "general_data.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            df_all.to_excel(writer, sheet_name='general data')
            statistical_df.to_excel(writer, sheet_name="statistical_results")
    else:
        print(statistical_df)


def main_validation(path=None, save_images=True):
    # automate segmentation
    if path is None:
        path = uf.select_location("choose the directory of the dicom file")
    # Perform preprocessing on the medical images
    slices, images_for_sagittal, images, spacing, reverse = pre.preprocessing(path)
    # Perform region of interest (ROI) selection on the images
    slices, all_head, images_for_sagittal, images, edge_index, end_open_nose_index, naso_index, \
        end_nasopharynx, image_type, interference = roi.voi(slices, images_for_sagittal, images)
    # Perform nasal airway segmentation on the images
    slices, all_head, roi_images, seg_images, fix_seg_images, open_nostril_index, edge_index = \
        segment.seg_nasal_airway(slices, all_head, images, edge_index, end_open_nose_index, naso_index,
                                 end_nasopharynx, image_type)

    # (thickness- axial spacing, spacing[1]- coronal spacing, spacing[2] - sagittal spacing)
    vol_factor = spacing[0] * spacing[1] * spacing[2]

    # manual segmentation
    manual_segmentation = validation.exporting_nrrd()

    # validation
    validation.test(ground_truth=manual_segmentation, auto_segmentation=fix_seg_images, vol_factor=vol_factor)

    if save_images is True:
        presentation_images, pixels_data, axial_data, coronal_data = validation.compare_segmentation(
            slices=slices, roi_images=roi_images, segmentation=fix_seg_images, ground_truth=manual_segmentation)
        presentation.save_img(presentation_images, mode="compare_to_manual", anim=False)


class MainApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Neonatal Congenital Nasal Obstruction Analysis")

        # Button to run Segmentation and Measurements
        self.button_segmentation = tk.Button(master, text="1. Segmentation and Measurements", command=self.run_segmentation)
        self.button_segmentation.pack()

        # Button to run Validation
        self.button_validation = tk.Button(master, text="2. Validation", command=self.run_validation)
        self.button_validation.pack()


        # Button to run General Data Calculation and Statistical Analysis
        self.button_general_data = tk.Button(master, text="3. General Data Calculation and Statistical Analysis", command=self.run_general_data)
        self.button_general_data.pack()

        # Button to run Region Averaging and Statistical Analysis
        self.button_region_averaging = tk.Button(master, text="4. Region Averaging and Statistical Analysis", command=self.run_region_averaging)
        self.button_region_averaging.pack()


        # Button to run Standardization and Visualization
        self.button_standardization = tk.Button(master, text="5. Standardization and Visualization", command=self.run_standardization)
        self.button_standardization.pack()

        # Button to Exit the application
        self.button_exit = tk.Button(master, text="0. Exit", command=self.master.destroy)
        self.button_exit.pack()

    def run_segmentation(self):
        path = self.select_path("Select Case Location")
        # Check if the selected directory contains DICOM files
        # dicom_files = [file for file in os.listdir(path) if file.lower().endswith('.dcm')]
        # if not dicom_files:
        #    messagebox.showerror("Error", "No DICOM files found in the selected folder. Please choose a valid directory.")
        #    return
        save_path = self.select_save_path()
        os.chdir(save_path)

        # User choices
        save_data = self.ask_yes_no("Do you want to save data?")

        saving_axial = self.ask_yes_no("Do you want to save axial images?")
        if saving_axial is True:
            axial_anim = self.ask_yes_no("Do you want to create axial animations?")
        else:
            axial_anim = False
        saving_coronal = self.ask_yes_no("Do you want to save coronal images?")
        if saving_coronal is True:
            coronal_anim = self.ask_yes_no("Do you want to create coronal animations?")
        else:
            coronal_anim = False

        model3d = self.ask_yes_no("Do you want to generate 3D models?")
        plot_measure = self.ask_yes_no("Do you want to plot measurements?")
        pa_width = self.ask_yes_no("Do you want to calculate PA width?")

        # Add logic for running Segmentation and Measurements here
        main_measurement(path=path, save_path=save_path, saving_coronal=saving_coronal,
                         saving_axial=saving_axial, model3d=model3d, plot_measure=plot_measure,
                         save_data=save_data, pa_width=pa_width, axial_anim=axial_anim, coronal_anim=coronal_anim)
        print("Completed")

    def run_standardization(self):
        path = self.select_path()

        # User choices
        save_data = self.ask_yes_no("Do you want to save data?")

        # Choose specific cases or use default
        cases = self.select_cases("Enter specific cases for standardization (comma-separated) or press Enter for default:", default_cases=list(range(0, 29)))

        # Choose specific exclude cases or use default
        exclude_cases = self.select_cases("Enter specific cases to exclude (comma-separated) or press Enter for default (18, 19, 28):", default_cases=(18, 19, 28))

        # Add logic for running Standardization and Visualization here
        main_standardization_3_all(cases=cases, exclude_cases=exclude_cases, saving_data=save_data, path=path)
        print("Completed")

    def run_region_averaging(self):
        path = self.select_path()

        # User choices
        save_data = self.ask_yes_no("Do you want to save data?")
        ttest = self.ask_yes_no("Do you want to perform t-test analysis?")

        # Choose specific cases or use default
        cases = self.select_cases("Enter specific cases for region averaging (comma-separated) or press Enter for default:", default_cases=list(range(0, 29)))

        # Choose specific exclude cases or use default
        exclude_cases = self.select_cases("Enter specific cases to exclude (comma-separated) or press Enter for default (18, 19, 28):", default_cases=(18, 19, 28))

        # Add logic for running Region Averaging and Statistical Analysis here
        main_region_averaging(cases=cases, path=path, exclude_cases=exclude_cases,
                              saving_data=save_data, ttest=ttest)
        print("Completed")

    def run_general_data(self):
        path = self.select_path()

        # User choices
        save_data = self.ask_yes_no("Do you want to save data?")
        ttest = self.ask_yes_no("Do you want to perform also t-test analysis?")

        # Choose specific cases or use default
        cases = self.select_cases("Enter specific cases for general data calculation (comma-separated) or press Enter for default:", default_cases=list(range(0, 29)))

        # Choose specific exclude cases or use default
        exclude_cases = self.select_cases("Enter specific cases to exclude (comma-separated) or press Enter for default (18, 19, 28):", default_cases=(18, 19, 28))

        # Add logic for running General Data Calculation and Statistical Analysis here
        main_general_data(cases=cases, path=path, exclude_cases=exclude_cases,
                          saving_data=save_data, ttest=ttest)
        print("Completed")

    def run_validation(self):
        path_for_auto = self.select_path("select the folder for the automate segmentation")


        # User choices
        save_images = self.ask_yes_no("Do you want to save images?")

        # Add logic for running Validation here
        main_validation(path=path_for_auto, save_images=save_images)

        print("Completed")

    def select_path(self, title="select the location of the measurements folder"):
        path = filedialog.askdirectory(title=title)
        while not os.path.isdir(path):
            messagebox.showerror("Error", "Invalid directory. Please select an invalid directory.")
            path = filedialog.askdirectory(title=title)
        return path

    def select_save_path(self, title="Select Save Location"):
        path = filedialog.askdirectory(title=title)
        if not os.path.isdir(path):
            messagebox.showerror("Error", "Invalid directory. Please select a valid folder.")
            path = filedialog.askdirectory(title=title)
        return path

    def ask_yes_no(self, question):
        response = messagebox.askyesno("Question", question)
        return response

    def select_cases(self, prompt, default_cases):
        user_input = input(f"{prompt} (Default: {default_cases}): ")
        if user_input:
            try:
                cases = [int(case.strip()) for case in user_input.split(',')]
                # Limit cases to the range [0, 28]
                cases = [case for case in cases if 0 <= case <= 28]
                return cases
            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please enter comma-separated integers.")
                return self.select_cases(prompt, default_cases)
        else:
            return default_cases


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
   # main_measurement(path=r"C:\Users\owner\Desktop\cases\#6", save_path=r"C:\Users\owner\Downloads\try", saving_coronal=False,
    #                     saving_axial=False, model3d=False, plot_measure=False,
     #                    save_data=False, pa_width=True, axial_anim=False, coronal_anim=False)



