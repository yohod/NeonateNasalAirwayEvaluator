import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preprocessing as pre
import roi
import usefull_function as uf
import segmentation as segment
import presentation
import measurment as msure
import model_3d as m3d
import measurement_visulization as visul
import standardization as standart
import bone_distance as bone
import validation
import tkinter as tk
from tkinter import filedialog

def select_location(title):
    path = filedialog.askdirectory(title=title)
    return path

# 1. segmentation, save images, measurement, plot them and save them, 3d model, bone distance, (timeing?)
#
# 2. save general data, save csa in specific regions, plot CSA standtardization
# 3. validation an


def main_standardization_3_all():
    # Load all Excel dataframes
    path = "C:\\Users\\owner\\Desktop\\cases\\data\\"
    normal_list = []
    obstruct_list = []
    surgery_list = []
    pa_percent_list = []
    ch_percent_list = []

    for mode in ["all", "2connected", "1"]:
        if mode == "1" or mode == "2connected":
            continue
        for i in range(0, 29):
            print(i)
            if i in [1,2,9, 24] or i in [18, 19, 28]:  # 10,11,
                continue
            exel_path = path + uf.CASE_MAP[str(i)] + "(" + str(i) + ")" + '.xlsx'

            df, pa_percent, ch_percent = standart.read(exel_path, mode=mode, inferior=False,
                                                       percent_method="percentage of nasal airway")
            pa_percent_list.append(pa_percent)
            ch_percent_list.append(ch_percent)
            if i in [0, 4, 6, 7, 10, 11]:
                surgery_list += df
            elif i in [3, 5, 8, 12]:
                obstruct_list += df
            else:
                normal_list += df

    all_lists = [normal_list, obstruct_list, surgery_list, ]

    stand_df = []
    for df_list in all_lists:
        avg_df, std_df = standart.standardize_cases(df_list, percent_col_name="percentage of nasal airway")
        stand_df.append([avg_df, std_df])

    saving_data = False
    if saving_data:
        excel_path = "C:\\Users\\owner\\Desktop\\cases\\data\\obstruct_averaging_all" + '(' + mode + ')' + '.xlsx'
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
    print(pa_percent_list,np.mean(pa_percent_list) )
    pa_percent = round(np.mean(pa_percent_list),1)
    ch_percent = round(np.mean(ch_percent_list),1)
    standart.plot_compare_3(stand_df, std_flag=std_flag, mode=mode, percent_mode="all",
                            pa_percent=pa_percent, ch_percent=ch_percent,units ="mm")


def main_region_averaging():
        # load all exel dataframe
    path = "C:\\Users\\owner\\Desktop\\cases"
    df_all = pd.DataFrame(columns=['case','region', 'vol', 'avg area', 'std', 'diagnose'])

    for i in range(0, 29):
        print(i)
        if i in [1,2,9, 24]:
            continue
        slices, images, thickness, reverse = pre.load_dicom(path + "\\#" + str(i))
        exel_path = path + "\\data\\" + uf.CASE_MAP[str(i)] + "(" + str(i) + ")" + '.xlsx'
        if i in [0,4, 6, 7, 10, 11]:
            diagnose = 'CNPAS + surgery'
        elif i in [3, 5, 8, 12]:
            diagnose = 'CNPAS WO surgery'
        else:
            diagnose = 'normal'

        for region in ["PA", "25%", "50%", "75%", "CH"]:
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
            vol, avg, std = standart.average(exel_path, slices[3].PixelSpacing[0], inferior=False,
                                             min_percent=min_percent, max_percent=max_percent)
            case_data = pd.DataFrame(
                    {'case': [i], 'region':[region], 'vol': [vol], 'avg area': [avg], 'std': [std], 'diagnose': [diagnose]})

            df_all = pd.concat([df_all, case_data], ignore_index=True)

        saving_data = True
        if saving_data is True:
            excelpath = "C:\\Users\\owner\\Desktop\\cases\\data\\" + "averaging_region_area.xlsx"
            with pd.ExcelWriter(excelpath) as writer:
                for region in ["PA", "25%", "50%", "75%", "CH"]:
                    df_all[df_all['region']== region].to_excel(writer, sheet_name=region)



def main_genral_data():
    # load all exel dataframe
    path = "C:\\Users\\owner\\Desktop\\cases"
    df_all = pd.DataFrame(
        columns=['case', 'vol', 'not c vol', 'surface area', 'vol nostrils', 'vol internal', 'vol naso',
                 'not c nostrils', 'not c internal', 'inferior vol', 'inferior not cc vol', 'not c naso',
                 'min cs area nares', 'percent of min nares', 'min cs area internal', 'percent of min internal',
                 'min cs area pa', 'percent of min pa',
                 'len all', 'len pa-ch', 'diagnose']
    )
    df_all = df_all.set_index('case')
    for i in range(0, 29):
        print(i)
        if i in [1,2,9, 24]:
            continue
        # slices = pre.load_dicom(path + "\\#" + str(i))
        exel_path = path + "\\data\\" + uf.CASE_MAP[str(i)] + "(" + str(i) + ")" + '.xlsx'
        if i in [4, 6, 7, 10, 11]:
            diagnose = 'CNPAS + surgery'
        elif i in [3, 5, 8, 12]:
            diagnose = 'CNPAS WO surgery'
        else:
            diagnose = 'normal'
        vol_df = pd.read_excel(exel_path, sheet_name="cc volume")
        not_c_vol_df = pd.read_excel(exel_path, sheet_name="not cc volume")
        inferior_df = pd.read_excel(exel_path, sheet_name="inferior cc volume")
        not_cc_inferior_df = pd.read_excel(exel_path, sheet_name="inferior not cc volume")
        case_data = pd.DataFrame({'case': [i], 'vol': vol_df.iloc[0]["cc total volume"],
                                  'not c vol': not_c_vol_df.iloc[0]["not_cc total volume"],
                                  'surface area': vol_df.iloc[0]["surface area"],
                                  'vol nostrils': vol_df.iloc[0]["nostrils volume"],
                                  'vol internal': vol_df.iloc[0]["pa-ch volume"],
                                  'vol naso': vol_df.iloc[0]["nasopharynx volume"],
                                  'not c nostrils': not_c_vol_df.iloc[0]["not_cc nostrils volume"],
                                  'not c internal': not_c_vol_df.iloc[0]["not_cc pa-ch volume"],
                                  'inferior vol': inferior_df.iloc[0]["vol r"] + inferior_df.iloc[0]["vol l"],
                                  'inferior not cc vol': not_cc_inferior_df.iloc[0]["vol r"] +
                                        not_cc_inferior_df.iloc[0]["vol l"],
                                  'not c naso': not_c_vol_df.iloc[0]["not_cc nasopharynx volume"],
                                  'min cs area nares': vol_df.iloc[0]["min cs area nares"],
                                  'percent of min nares': vol_df.iloc[0]["percent of min nares"],
                                  'min cs area internal': vol_df.iloc[0]["min cs area internal"],
                                  'percent of min internal': vol_df.iloc[0]["percent of min internal"],
                                  'min cs area pa' :  vol_df.iloc[0]["min cs area pa"],
                                  'percent of min pa': vol_df.iloc[0]["percent of min pa"],
                                  'len all': vol_df.iloc[0]["length"],
                                  'len pa-ch': vol_df.iloc[0]["pa-ch length"], 'diagnose': [diagnose]})
        case_data = case_data.set_index('case')
        df_all = pd.concat([df_all, case_data], ignore_index=False)

    df_surgery = df_all[df_all["diagnose"] == 'CNPAS + surgery']
    df_cnpas = df_all[df_all["diagnose"] == 'CNPAS WO surgery']
    df_normal = df_all[df_all["diagnose"] == 'normal']


    saving_data = True
    if saving_data is True:
        excelpath = "C:\\Users\\owner\\Desktop\\cases\\data\\" + "general_data.xlsx"
        with pd.ExcelWriter(excelpath) as writer:
            df_all.to_excel(writer, sheet_name='general data')






def main_mesurement(path ="",save_path="", plot_measure=True, model3d=True, saving_axial=True,
                    axial_anim=False, saving_coronal=True, coronal_anim=False, save_data = True,
                    pa_width=True):
    if path == "":
        path = select_location("Select Case Location")
    if save_path == "":
        save_path = select_location("Select Save Location")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    os.chdir(save_path)

    slices, images_for_sagittal, images,spacing, reverse = pre.preprocessing(path)

    slices, all_head, images_for_sagittal, images, edge_index, end_open_nose_index, naso_index, end_nasopharynx, \
    image_type, interferent = roi.voi(slices, images_for_sagittal, images)
    slices, all_head, roi_images, seg_images, fix_seg_images, open_nostril_index, edge_index = \
                                        segment.seg_nasal_airway(slices, all_head, images,
                                        edge_index, end_open_nose_index, naso_index, end_nasopharynx, image_type)
    # to change the global threshold or the local threshold boundaries, replace the last code line with other HU values
    # slices, all_head, roi_images, seg_images, fix_seg_images, open_nostril_index, edge_index =
    # seg_nasal_airway(slices, all_head, images, edge_index, end_open_nose_index, naso_index,
    #                  end_nasopharynx, image_type, global_thresh = -400 , local_thresh(-400 ,-125))


    thickness = spacing[0]
    spacing = spacing[1:]

    aperture_index = msure.find_aperture(roi_images, edge_index, image_type)

    choana_index = msure.find_choana(fix_seg_images, roi_images)
    if saving_axial is True:
        axial_images = presentation.presentation(slices, roi_images, seg_images, fix_seg_images,
                                                 plane_mode="axial", additional_slices= all_head)
        presentation.save_img(axial_images, save_path=save_path, mode='axial', anim=axial_anim)
    if saving_coronal is True:
        coronal_images = presentation.presentation(slices, roi_images, seg_images, fix_seg_images,
                                                       plane_mode="coronal")
        presentation.save_img(coronal_images, save_path=save_path, mode='coronal', anim=coronal_anim)
    if model3d is True:
        m3d.reconstruction3d(fix_seg_images[:,:,:], spacing, thickness, save_path=save_path, connected=True)
    measurement_data = msure.measurement(fix_seg_images, aperture_index, choana_index, thickness, spacing,
                                            image_type)
    if plot_measure is True:
        visul.plot(measurement_data[0], aperture_index, choana_index, save_path=save_path)
        visul.plot_inferior(measurement_data[1], aperture_index, choana_index, save_path=save_path)

    if save_data is True:
        excelpath = save_path +"\\" + 'measurements.xlsx'
        sheet_name = [
            ['cc volume', 'not cc volume', 'cs r data', 'cs l data', 'notcc cs r data', 'notcc cs l data'],
            ['inferior cc volume', 'inferior not cc volume', 'inferior cs r data', 'inferior cs l data',
             'inferior notcc cs r data', 'inferior notcc cs l data']]
        with pd.ExcelWriter(excelpath) as writer:
            for i in range(0, 12):
                mode_index = i // 6
                data_index = i % 6
                measurement_data[mode_index][data_index].to_excel(writer,
                                                                 sheet_name=sheet_name[mode_index][data_index])
    if pa_width is True:
        bone.measure_bone_distance(fix_seg_images, roi_images, aperture_index, choana_index, spacing,
                                   thickness, save_path=save_path, hu_scale=True)


# the additional operation need to be updated for using
def main_compare():
    path = r"C:\Users\owner\Desktop\cases\#"
    case_df = []
    pa_indexs = []
    ch_indexs = []
    # case_input = input("enter 2 cases to compare from 3 to 27 (exlude 24) space-separted ")
    # case1, case2 = tuple(case for case in case_input.split())
    case1, case2 = '3', '14'
    cases_path = uf.CASE_MAP[case1] + " Via " + uf.CASE_MAP[case2]
    case_label = [uf.CASE_MAP[case1], uf.CASE_MAP[case2]]
    for i in [case1, case2]:
        thresh = 650
        slices, images_for_sagittal, images, reverse = pre.preprocessing(path + i)
        slices, all_head, images_for_sagittal, images, edge_index, end_open_nose_index, end_nasopharynx, image_type, \
            interferent = roi.roi(slices, images_for_sagittal, images)

        slices, all_head, roi_images, seg_images, seg_images2, fix_seg_images, fix_seg_images2, open_nostril_index = \
            segment.seg_nasal_airway(slices, all_head, images, thresh, edge_index, end_open_nose_index, end_nasopharynx,
                                 image_type)
        spacing, thickness = slices[0].PixelSpacing, slices[0].SliceThickness
        aperture_index = msure.find_aperture(roi_images, edge_index)
        pa_indexs.append(aperture_index)
        choana_index = msure.find_choana(fix_seg_images, roi_images, end_open_nose_index, end_nasopharynx)
        ch_indexs.append(choana_index)
        measurment_data = msure.measurement(fix_seg_images, aperture_index, choana_index, thickness, spacing)
        case_df.append(measurment_data)

    visul.plot_compare3(case_df, pa_indexs, ch_indexs, case_label, cases_path)


def main_compare3():
    path = r"C:\Users\owner\Desktop\cases\#"
    cases_df = []
    cases_inferior_df = []
    pa_indexs = []
    ch_indexs = []
    # case_input = input("enter 2 cases to compare from 3 to 27 (exlude 24) space-separted ")
    # case1, case2 = tuple(case for case in case_input.split())
    cases = ['21', '3', '6']
    cases_type = [1, 2, 3]
    cases_path = ""
    for case in cases:
        cases_path += case
        if case != cases[-1]:
            cases_path += " Via "

    for i in cases:
        thresh = 650
        slices, images_for_sagittal, images, reverse = pre.preprocessing(path + i)
        slices, all_head, images_for_sagittal, images, edge_index, end_open_nose_index, end_nasopharynx, image_type, \
            interferent = roi.roi(slices, images_for_sagittal, images)

        slices, all_head, roi_images, seg_images, fix_seg_images, open_nostril_index, _ = \
            segment.seg_nasal_airway(slices, all_head, images, thresh, edge_index, end_open_nose_index, end_nasopharynx,
                                  image_type)
        spacing, thickness = slices[0].PixelSpacing, slices[0].SliceThickness
        aperture_index = msure.find_aperture(roi_images, edge_index)
        pa_indexs.append(aperture_index)
        choana_index = msure.find_choana(fix_seg_images, roi_images, end_open_nose_index, end_nasopharynx)
        ch_indexs.append(choana_index)
        measurment_data = msure.measurement(fix_seg_images, aperture_index, choana_index, thickness, spacing)
        cases_df.append(measurment_data[0])
        cases_inferior_df.append(measurment_data[1])

    visul.plot_compare3(cases_df, cases_type, pa_indexs, ch_indexs, "x", cases, cases_path)
    visul.plot_compare3_inferior(cases_inferior_df, cases_type, "x", cases, cases_path)





def main_save_cornal():
    path = r"C:\Users\owner\Desktop\cases\#"
    for i in range(9, 29):
        if i == 24:
            continue
        print(uf.CASE_MAP[str(i)], "(", i, ")")
        thresh = 650

        slices, images_for_sagittal, images, reverse = pre.preprocessing(path + str(i))
        slices, all_head, images_for_sagittal, images, edge_index, end_open_nose_index, end_nasopharynx, image_type, \
            interferent = roi.roi(slices, images_for_sagittal, images)
        # bone_seg(images, edge_index, end_open_nose_index)
        slices, all_head, roi_images, seg_images, seg_images2, fix_seg_images, fix_seg_images2, open_nostril_index = \
            segment.seg_nasal_airway(slices, all_head, images, thresh, edge_index, end_open_nose_index, end_nasopharynx,
                                 image_type)
        aperture_index = msure.find_aperture(roi_images, edge_index)
        choana_index = msure.find_choana(fix_seg_images, roi_images, end_open_nose_index, end_nasopharynx)
        mid_pa, mid_ch = msure.finding_inferior_conca(fix_seg_images, aperture_index, choana_index)
        case = "#" + uf.CASE_MAP[str(i)] + "(" + str(i) + ")"
        savepath = "C:\\Users\\owner\\Desktop\\cases\\coronal_images\\" + case
        # crnl.save_coronal_bone(roi_images, aperture_index, choana_index, savepath)
        # crnl.searching_mid_tur(fix_seg_images, roi_images, aperture_index, choana_index,
        # mid_pa,mid_ch, savepath, case)



def main_valdiation():
    path = r"C:\Users\owner\Desktop\cases\#"
    for i in [ 6, 24,14, 17]:  # [10,11]:
        print(uf.CASE_MAP[str(i)], "(", i, ")")
        thresh = 624
        newpath = r"C:\Users\owner\Desktop\cases\roi_dicom" + "\\" + str(i)
        if i == 10 or i == 11:
            ground_truth = validation.exporting_nrrd(newpath, "Segmentation.seg.nrrd", True)
        # continue
        slices, images_for_sagittal, images,thickness, reverse = pre.preprocessing(path + str(i))
        slices, all_head, images_for_sagittal, images, edge_index, end_open_nose_index, naso_index, end_nasopharynx, image_type, \
        interferent = roi.roi(slices, images_for_sagittal, images)
        slices, all_head, roi_images, seg_images, fix_seg_images, open_nostril_index, edge_index = \
            segment.seg_nasal_airway(slices, all_head, images, thresh, edge_index, end_open_nose_index, naso_index,
                                  end_nasopharynx,
                                  image_type)

        # validation.update_image_to_dicom(slices,roi_images,newpath)
        vol_factor = slices[1].PixelSpacing[0] * slices[1].PixelSpacing[1] * thickness
        aperture_index = msure.find_aperture(roi_images, edge_index,image_type)
        choana_index = msure.find_choana(fix_seg_images, roi_images, end_open_nose_index, end_nasopharynx)
        end = aperture_index + int(0.10 * (choana_index - aperture_index))

        # validation.test(newpath, np.swapaxes(fix_seg_images,0,2),aperture_index,choana_index, "Segmentation.nrrd")
        ground_truth = validation.exporting_nrrd(newpath, "Segmentation.seg.nrrd")
        swap_seg = np.swapaxes(fix_seg_images, 0, 2)
        ground_truth = validation.tikon(ground_truth, swap_seg)
        validation.test(newpath, swap_seg, vol_factor=vol_factor)
        print("pa-ch")
        validation.test(newpath, swap_seg, start=aperture_index, end=choana_index, vol_factor=vol_factor)
        #compare_images, g_data, axial_data, coronal_data = validation.compare_segmentation(slices, roi_images,
        #                                                                                   swap_seg, ground_truth,
        #                                                                                   aperture_index, choana_index)
        #print(g_data)
        #spacing = round(slices[1].PixelSpacing[0], 3)
        #spacing **= 2
        #g_data = np.round(np.array(g_data) * spacing / (choana_index - aperture_index), 2)
        #print(g_data)
        print("pa region")
        validation.test(newpath, swap_seg, start=aperture_index, end=end, vol_factor=vol_factor)
        compare_images, g_data, axial_data, coronal_data = validation.compare_segmentation(slices, roi_images,
                                                                                           swap_seg, ground_truth,
                                                                                           aperture_index, choana_index)
        #print(g_data)
        spacing = round(slices[1].PixelSpacing[0], 3)
        spacing **= 2
        #g_data = np.round(np.array(g_data) * spacing / (end - aperture_index), 2)
        #print(g_data)
        validation.save_imgs(compare_images,i , spacing)

        # print(spacing)
        #spacing **= 2
        # print (g_data)
        # print("axial------>")
        # for i in range(len(axial_data[0])):
        #   break
        #   print (i, ":", axial_data[0][i],axial_data[1][i], end = "| ")
        # print()
        # print("############################################")
        # print("coronal------>")
        # for i in range(len(coronal_data[0])):
        #   break
        #  print(i, ":", coronal_data[0][i], coronal_data[1][i], end="| ")
        # print()
        #print("############################################")
        #print("inferior result")
        #pa_index, ch_index = measurment.finding_inferior_conca(fix_seg_images, aperture_index, choana_index, image_type)
        #row_index = min(min(pa_index), min(ch_index))
        #ground_truth = validation.tikon(ground_truth[:, :, row_index:], swap_seg[:, :, row_index:])
        #validation.test(newpath, swap_seg, inferior=row_index)
        #print("pa-ch inferior")
        #validation.test(newpath, swap_seg, start=aperture_index, end=choana_index, inferior=row_index)
        #print("pa region inferior")
        #validation.test(newpath, swap_seg, start=aperture_index, end=end, inferior=row_index)

        # print ("result:",quality.quality_test(roi_images, fix_seg_images))
        # spacing, thickness = slices[0].PixelSpacing, slices[0].SliceThickness
        # aperture_index = msure.find_aperture(roi_images, edge_index)
        # choana_index = msure.find_choana(fix_seg_images, roi_images, end_open_nose_index, end_nasopharynx)
        # axial_images = presentation.presentation(slices, roi_images, seg_images, fix_seg_images,plane_mode = "axial")
        # presentation.save_img(axial_images, i, mode='axial', anim=True)
        # coronal_images = presentation.presentation(slices, roi_images, seg_images, fix_seg_images, plane_mode="coronal")
        # presentation.save_img(coronal_images, i, mode='coronal', anim=True)


def main_image():
    import time
    new_path = r"C:\Users\owner\Downloads\example"
    path = r"C:\Users\owner\Desktop\cases\#"
    timer1 = []
    timer2 = []
    timer3 = []
    timer4 = []

    for i in range(27, 28):  # range(5,21):# [11,26,10,4,7,17,21,23]:#
        if i in [9, 10, 11, 24]:
            continue
        # print(uf.CASE_MAP[str(i)], "(", i, ")")
        print(i)
        thresh = 624
        time0 = time.time()
        slices, images_for_sagittal, images, reverse = pre.preprocessing(path + str(i))
        # slices, images_for_sagittal, images, reverse = pre.preprocessing(new_path + str(i), old=True)

        slices, all_head, images_for_sagittal, images, edge_index, end_open_nose_index, naso_index, end_nasopharynx, image_type, \
        interferent = roi.roi(slices, images_for_sagittal, images, old=False)
        # newpath = r"C:\Users\owner\Desktop\cases\roi_dicom" + "\\" + str(i)
        # validation.update_image_to_dicom(slices, images, newpath= newpath)
        slices, all_head, roi_images, seg_images, fix_seg_images, open_nostril_index, edge_index = \
            segment.seg_nasal_airway(slices, all_head, images, thresh, edge_index, end_open_nose_index, naso_index,
                                  end_nasopharynx, image_type)
        time1 = time.time()
        axial_images = presentation.presentation(slices, roi_images, seg_images, fix_seg_images, plane_mode="axial")
        #presentation.save_img(axial_images, i, mode='axial', anim=False)
        coronal_images = presentation.presentation(slices, roi_images, seg_images, fix_seg_images, plane_mode="coronal")
        #presentation.save_img(coronal_images, i, mode='coronal', anim=False)

        # import model_3d as model
        # model.reconstruction3d(fix_seg_images,slices[0].PixelSpacing, slices[0].SliceThickness, i)

        pa_index = msure.find_aperture(images, edge_index, image_type)
        if i == 4:
            fix_seg_images[35:, :pa_index, :] = 0
        # bone.mid_turbinate(roi_images,fix_seg_images,pa_index)
        ch_index = msure.find_choana(fix_seg_images, roi_images, end_open_nose_index, end_nasopharynx)

        spacing, thickness = slices[0].PixelSpacing, slices[0].SliceThickness
        m3d.reconstruction3d(fix_seg_images, slices[0].PixelSpacing, slices[0].SliceThickness)
        # savepath = "C:\\Users\\owner\\Desktop\\cases\\bone\\" + str(i)
        # bone.measure_bone_distance(fix_seg_images, roi_images, pa_index, ch_index, edge_index, spacing,
        #                          thickness, savepath)
        time2 = time.time()
        measurment_data = msure.measurement(fix_seg_images, pa_index, ch_index, thickness, spacing, image_type)
        visul.plot(measurment_data[0], pa_index, ch_index, pathcase=i)
        visul.plot_inferior(measurment_data[1], pa_index, ch_index, not_cc=False, pathcase=i)
        time3 = time.time()
        timer2.append(round(time2 - time1, 2))
        timer3.append(round(time3 - time2, 2))

        saving_data = False
        if saving_data is True:
            excelpath = "C:\\Users\\owner\\Desktop\\cases\\data\\" + uf.CASE_MAP[str(i)] + "(" + str(i) + ")" + '.xlsx'
            sheet_name = [
                ['cc volume', 'not cc volume', 'cs r data', 'cs l data', 'notcc cs r data', 'notcc cs l data'],
                ['inferior cc volume', 'inferior not cc volume', 'inferior cs r data', 'inferior cs l data',
                 'inferior notcc cs r data', 'inferior notcc cs l data']]
            with pd.ExcelWriter(excelpath) as writer:
                for j in range(0, 12):
                    mode_index = j // 6
                    data_index = j % 6
                    measurment_data[mode_index][data_index].to_excel(writer,
                                                                     sheet_name=sheet_name[mode_index][data_index])

    print(timer2)
    print(timer3)




if __name__ == '__main__':

    #main_visual()

    main_mesurement(path=r"C:\Users\owner\Desktop\cases\#6",save_path=r"C:\Users\owner\Desktop\try", saving_coronal=False,
                    saving_axial=False, model3d=False, plot_measure=False, save_data=False, pa_width=False)

    #main_standardization_3_all()
    # main_compare3()
    # main_bone()

    # main_coronal_segmentation()

    #main_valdiation()
    #main_genral_data()

    #main_region_averaging()
    # main_compare3_for_poster()
    # main_image()

