import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import usefull_function as uf
import os
import cc3d


#
def measure_bone_distance(seg_images, roi_images, pa_index, ch_index, spacing, thickness, save_path, hu_scale = False):
    """

    :param seg_images: the airway images segmented from the roi
    :param roi_images: the roi images with all the pixel data
    :param pa_index: the index of pyriform aperture
    :param ch_index: the index of the choana
    :param spacing: pixel spacing (coronal, sagittal)
    :param thickness: the axial spacing
    :param path: the saving path
    :return:
    """

    area_factor = thickness * spacing[1]
    width_factor = spacing[1]
    slice_data_list =[]
    BONE_THRESH = 1224 # the intercept usually is 1024' so in HU it is 200 HU
    if hu_scale is True:
        BONE_THRESH = 200


    # two dataframe the second contain only sections with PA bigger than 11 mm
    bone_distance_data = pd.DataFrame(columns=['coronal index','min row','min width', 'area', 'avg', 'begin col', 'end col'])
    bone_distance_data_more = pd.DataFrame(columns=['coronal index','min row','min width', 'begin col', 'end col', 'num row more 11'])

    # transform axial data to coronal

    coronal_roi = uf.axial_to_coronal(roi_images)
    coronal_seg = uf.axial_to_coronal(seg_images)
    cc_airway_label = uf.remove_unconnected_objects(coronal_seg)
    # label of the connected airway. needed for erasing row with one side bone + septum from being measured

    # segmenting the bones from coronal roi
    #
    coronal_bone1 = uf.binary_image(coronal_roi, BONE_THRESH) # 200 HU
    coronal_bone = cc3d.largest_k(coronal_bone1, 1)
    if False: # visualization
        for image in [coronal_roi[pa_index,:,:], coronal_bone1[pa_index,:,:],coronal_bone[pa_index,:,:] ]:
            plt.imshow(image, cmap=plt.cm.gray)
            plt.xticks([])
            plt.yticks([])
            plt.show()

     # index where to stop measuring bone distance. 15% of the midnasal length
    coronal_end_index = int(0.15 * (ch_index - pa_index)) + pa_index


    if False: # visulaztion
        img = uf.gray_to_color(roi_images[edge_index])
        #print(edge_index, coronal_end_index-pa_index)

        img[pa_index,:] = (255,0,0)
        img[coronal_end_index,:] = (255,0,0)
        #img [pa_index+add,:] = (255,255,0)
        plt.imshow(img)
        plt.show()

    # the end axial row to be measure

    interior_images = []
   # segmenting the internal region between the bones
    for coronal_index in range(pa_index, coronal_end_index-1): #
        image = coronal_bone[coronal_index]
        roi_image = coronal_roi[coronal_index]
        interior_image = interior_bones(image,roi_image, cc_airway_label[coronal_index])
        interior_images.append(interior_image)



        if False: # visualization
            if path !="":
                if not os.path.exists(path):
                    os.mkdir(path)
                os.chdir(path)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(roi_image, cmap=plt.cm.gray)
            ax2.imshow(image, cmap=plt.cm.gray)
            ax3.imshow(interior_image, cmap=plt.cm.gray)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.show()
            plt.savefig(str(coronal_index) + ".jpg")
            plt.close("all")


    #find low row of the PA
    interior_images = np.array(interior_images)
    for i in range(10):
        image = interior_images[i]
        if np.count_nonzero(image) > 200:
            end_row = np.nonzero(image)[0][-1]
            break

    #plot_interior_axial(coronal_roi, interior_images, pa_index, coronal_end_index,end_row, path)

    # measuring bone distance
    for coronal_index in range(pa_index, coronal_end_index-1):

        interior_image = interior_images[coronal_index-pa_index,:,:]

        # measure
        slice_data,_ = slice_bone_distance(interior_image, end_row)
        slice_data_list.append(slice_data)

        #floor_axial = min(edge_index+5, end_row). some cases need a floor limitation
        #slice_data_more = slice_data[(slice_data['width'] > 11 / width_factor) & (slice_data['row'] < floor_axial) & (slice_data['row'] > 15)]
        slice_data_more = slice_data[(slice_data['width'] > 11/ width_factor) & (slice_data['row'] > 15)]
        if not slice_data_more.empty:
            low_row = slice_data_more.iloc[-1]
            for image in [uf.np_to_color(interior_image), uf.gray_to_color(coronal_roi[coronal_index])]:
                if False:
                    break
                for i in range(0,image.shape[1],4):
                    image[14:16,i:i+3] = (255,128,0)
                image[low_row['row'],low_row["begin"]+1:low_row["end"]] = (0,255,0)

            data = pd.DataFrame({'coronal index': [coronal_index], 'min row': [low_row['row']],
                                 'min width': [round(low_row['width'] * width_factor, 2)],
                                 'begin col': [low_row["begin"]],
                                 'end col': [low_row["end"]],
                                 'num row more 11': [len(slice_data_more.index)]
                                 })
            bone_distance_data_more = pd.concat([bone_distance_data_more, data], ignore_index=True)

        area = slice_data['width'].sum()
        avg = slice_data['width'].mean()

        slice_data = slice_data[(slice_data['row'] > end_row - 8) &(slice_data['row'] < end_row - 2) &
                                (slice_data['width'] > 3 / width_factor) ]
        pd_rows = len(slice_data.index)


        if pd_rows  == 0:
            continue


        min_width_row = slice_data[slice_data['width'] == slice_data['width'].min()].iloc[-1]
        width = min_width_row['width']

        for image in [uf.np_to_color(interior_image), uf.gray_to_color(coronal_roi[coronal_index])]:
            for i in range(0,image.shape[1],4):
                image[end_row-10:end_row-8,i:i+3] = (255,128,0)

            image[min_width_row['row'],min_width_row["begin"]+1:min_width_row["end"]] = (255,0,0)

        data = pd.DataFrame({'coronal index': [coronal_index],'min row': [min_width_row['row']],
                'min width': [round( width * width_factor, 2)], 'area': [round(area * area_factor,2)],
                'avg': [round(avg * width_factor, 2)], 'begin col': [min_width_row["begin"]],
                'end col':[min_width_row["end"]]})
        bone_distance_data = pd.concat([bone_distance_data, data], ignore_index=True)

    #saving the PA width measure slice
    imshow_axail_bone_distance(coronal_roi, bone_distance_data, bone_distance_data_more)

    return bone_distance_data, slice_data_list




def interior_bones(image,roi_image, cc_airway_label):

    n,m = image.shape
    bone_image = np.copy(np.uint8(image))

    # whiting all the background
    for row in range(n):
        row_nonzero = np.nonzero(image[row])[0]
        row_nonzero_airway = np.nonzero(cc_airway_label[row])[0]
        # Selects first and last pixels of the bones
        # If you don't see an object whiten the whole row
        # If there is a bone on only one side whiten the whole row
        if len(row_nonzero) > 0:
            begin = row_nonzero[0]
            end = row_nonzero[-1]
        else:
            begin = m
            end = 0
# In the event that the bone is after the airway, it means that it is a nasal septum bone
# Or the airways are after the last bone, so as above
         # In these cases the line is ignored
         # anoter idea: search on the original image for the object by walking along the pixel row.
        # And according to the pixel value something that is both closest to the left of the respiratory tract and also has a value close to 200 Unisfeld units
        if len(row_nonzero_airway) > 0:
            if row_nonzero_airway[0] < begin:
                begin = row_nonzero_airway[0]
                begin = m
                # whitens all the row

            if row_nonzero_airway[-1] > end:
                end = 0
        # Whitens everything external to the bone
        bone_image[row,:begin] = 255
        bone_image[row,end:] = 255

    # inverse the image colores
    interior_region = np.copy(bone_image)
    interior_region[bone_image == 0] = 255
    interior_region[bone_image != 0] = 0
    interior_region1 = np.copy(interior_region)
    # remove low row objects
    # need to be improved
    num, label, stat, cent = uf.cc_stat(interior_region)
    for i in range(1,num):
        if i in label[n//2,:-n//2] and np.count_nonzero(label[n//4:-n//4,m//4:-m//4] == i) > 50:
            continue
        else:
            interior_region[label == i] = 0
    if False:
        fig, axis = plt.subplots(1, 4)
        axis[0].imshow(image, cmap=plt.cm.gray)
        axis[1].imshow(bone_image, cmap=plt.cm.gray)
        axis[2].imshow(interior_region1, cmap=plt.cm.gray)
        axis[3].imshow(interior_region, cmap=plt.cm.gray)
        for ax in axis:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()
    return interior_region

    
def slice_bone_distance(image, row_end):

    slice_data = pd.DataFrame(columns=['row', 'begin', 'end', 'width'])
    for row in range(0,row_end):
        row_nonzero = np.nonzero(image[row,:])[0]
        # if dont have two side pixels
        if len(row_nonzero) < 2:
            continue
        width = row_nonzero[-1] - row_nonzero[0] + 1
        row_data = pd.DataFrame({'row': [row], 'begin': [row_nonzero[0] - 1], 'end': [row_nonzero[-1] + 1],
                                 'width': [width]})
        slice_data = pd.concat([slice_data, row_data], ignore_index=True)

    if slice_data.empty:
        row_end_ = row_end
    else:
        row_end_ = row_data["row"].iloc[-1]

    return slice_data, row_end_

# for showing the PA width measurment on the slice image
def imshow_axail_bone_distance(images, bone_distance_data, bone_distance_data_more ):
    if bone_distance_data_more.empty :
        min_width = bone_distance_data["min width"].min()
        min_width_information = bone_distance_data[bone_distance_data["min width"] == min_width]
        min_row = min_width_information["min row"].max()
        min_width_information = min_width_information[min_width_information["min row"] == min_row]
    else:
        min_row = bone_distance_data_more["min row"].max()
        min_width_information = bone_distance_data_more[bone_distance_data_more["min row"] == min_row]
        min_width = min_width_information["min width"].min()
        min_width_information = min_width_information[min_width_information["min width"] == min_width]

    title = "Slice " + str(min_row)
    txt = str(min_width) + " mm"
    color = "green"
    if min_width < 11:
        color = "red"

        title += " With The Narrowest\n Nasal Bone's Distance.\n CNPAS Detected (< 11mm)"
    else:

        title += "\n Normal Bones Distance (> 11mm) "

    axial_image = images[:, min_row,:]
    axial_image = uf.gray_to_color(axial_image)
    axial_image[min_width_information["coronal index"].iloc[0],min_width_information["begin col"].iloc[0]] = (0,255,0)
    axial_image[min_width_information["coronal index"].iloc[0],min_width_information["end col"].iloc[0]] = (0,255,0)
    fig, ax = plt.subplots()
    ax.imshow(axial_image)
    xy1 = (min_width_information["begin col"].iloc[0], min_width_information["coronal index"].iloc[0])
    #xy1 = 40,59
    xy2 = (min_width_information["end col"].iloc[0], min_width_information["coronal index"].iloc[0])
    #xy2 = 55, 59

    ax.annotate("", xy=xy2, xytext=xy1, ha='left', arrowprops=dict(arrowstyle= "<->" ,color=color, shrinkA=0, shrinkB=0))

    xtxt =  (xy1[0] + xy2[0])//2
    ytxt = xy1[1] - 3
    ax.text(xtxt, ytxt ,txt, fontweight ="bold", color=color, ha='center', va='bottom')

    ax.set_title(title, color=color, fontweight ="bold")
    plt.tight_layout()
    plt.savefig("bone_distance.jpg")
    plt.close("all")




# for visulazation of the inernal region between the bones
def plot_interior_axial(coronal_roi_images, interior_images,begin_index,end_index, end_axial, savepath):
    savepath += "\\axial\\"
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    os.chdir(savepath)


    for axial_index in range(end_axial):
        roi_image = coronal_roi_images[:,axial_index,:]

        roi_image = uf.gray_to_color(roi_image)
        color_roi_image = np.copy(roi_image)
        color_area = np.copy(roi_image[begin_index:end_index, :])
        color_label = interior_images[:,axial_index,:]
        color_area[color_label != 0] = (255,128,0)
        color_roi_image[begin_index:end_index,:] = color_area
        color_roi_image[begin_index,:] = (255,0,0)
        color_roi_image[end_index, :] = (255, 0, 0)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.imshow(roi_image)
        ax2.imshow(color_roi_image)
        plt.tight_layout()

        plt.savefig(str(axial_index) + ".jpg")
        plt.close("all")






