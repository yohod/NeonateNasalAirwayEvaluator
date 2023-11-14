import os
import numpy as np
import nrrd
import usefull_function as uf
from matplotlib import pyplot as plt

# this module used to compare the automatic segmentation
# with manual segmentation made by the software 3d slicer





def quality_test(ground_truth, segmented_airway):
    """calculate Jaccard, Dice , sensitivity

    """

    X = (segmented_airway > 0)
    Y = (ground_truth > 0)
    X_and_Y = np.logical_and(X, Y)
    X_or_Y = np.logical_or(X, Y)
    X_nor_Y = np.logical_not(X_or_Y)

    J = round(100 * np.sum(X_and_Y) / np.sum(X_or_Y),2)
    D = round(200 * np.sum(X_and_Y) / (np.sum(X) + np.sum(Y)),2)


    FP = np.sum(np.logical_and(X,np.logical_not(Y)))#.astype(np.float)
    FN = np.sum(np.logical_and(Y,np.logical_not(X)))#.astype(np.float)
    TP = np.sum(X_and_Y)

    # to calculate accuracy
    TN = np.sum(X_nor_Y)
    num_of_pixels = X.size

    sensitivity = round(100 * TP / (TP + FN),2)

    return J, D ,sensitivity, (TP,FP,FN)

def exporting_nrrd(path, file_name, correct_dim=False):
    os.chdir(path)
    data, header = nrrd.read(file_name)
    if correct_dim:
        data = data.swapaxes(0,2)
    return data

# there were some cases without the same dimension,
# an update of the roi algorthm cause in some cause
# differents dimension of manual and automate segmentation
def tikon(data_nrrd,new_data):
    n1,m1,p1 = data_nrrd.shape
    n2,m2,p2 = new_data.shape
    new_matrix = np.zeros_like(new_data)
    if m2 - m1 >0:
        new_matrix[:,:m1,:]= data_nrrd
        new_matrix[:,m1:,:]=new_data[:,m1:,:]
    elif m2 - m1 <0:
        new_matrix = data_nrrd[:,:m2,:]

    if n2-n1 >0:
        new_matrix[:n1, :, :] = data_nrrd
        new_matrix[n1:, :, :] = new_data[n1:,:, :]
    elif n2 - n1 < 0:
        new_matrix = data_nrrd[n2:, :, :]

    return new_matrix

# for saving automate roi images and slices, for Simplify the manual segmentation process
def update_image_to_dicom(slices,b_slices, newpath = ""):
    """ save an update images and its size (rows & columns) at dicom files """
    if (newpath == "") :
        newpath = input("enter a path for the updated dicom files")
#crate a new folder in the computer
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(newpath)
    np_b_slices =[np.array(i).astype(np.uint16) for i in b_slices] #create a list of numpy array of 16-bit unsigned integer (sorted(, key= int() )
    images = [slices[i] for i in range(len(np_b_slices))] # taking the dicom source file
    for i in range(len(images)):
        images[i].Rows, images[i].Columns  = b_slices[i].shape # changing in the header the dicom image size
        images[i].PixelData = np_b_slices[i].tobytes() # pixel data is the data of the ct image in byte form. tobytes - transform a 16-bit unsigned integer numpy array to byetes        images[i].save_as(newpath + '\\' + str(i))
        images[i].save_as(str(i) + ".dcm")

def test(path, seg_images,start=0, end=-1,inferior=0,vol_factor=1, file_name ="Segmentation.seg.nrrd"):
    ground_truth = exporting_nrrd(path, file_name)
    ground_truth = tikon(ground_truth, seg_images)
    length = end - start
    if length <0 :
        length = np.shape(seg_images)[1]
    print(length)
    cs_factor = vol_factor/ length
    J, D ,sensitivity, (TP,FP,FN) = quality_test(ground_truth[:,start:end,inferior:],seg_images[:,start:end,inferior:], cs_factor)
    print ("fp:",FP*vol_factor, "fn:",FN*vol_factor)
###

# Saving the images that compare the manual and automatic segmentations.
# The identical pixels in both (TF) were marked in green.
# The pixels detected only in the manual segmentation (FN) were marked in red,
# And the pixels detected only in the automatic segmentation (FP) were marked in blue.

def compare_segmentation(slices, roi_images, segmentation, ground_truth, start, end):
    """

    :param slices: ct slices. only the voi slices that include airway
    :param roi_images: roi slices, and roi images
    :param segmentation: automate segmented airway
    :param ground_truth: manual segmented airway

    """

    # hu parameters for rescaling the image
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    X = np.copy(segmentation)
    Y = np.copy(ground_truth)

    tp = np.logical_and(X, Y).astype(int) * 255
    fp = np.logical_and(X,np.logical_not(Y)).astype(int) * 255
    fn = np.logical_and(Y,np.logical_not(X)).astype(int) * 255

    # data for understanding
    pixelsdata = [np.count_nonzero(x) for x in (tp[:,start:end,:], fp[:,start:end,:], fn[:,start:end,:])]
    axial_data = []
    coronal_data = []
    for x in (fp, fn):
        x = x[:, start:end + 1, :]
        slices_pixel_data_temp = [np.count_nonzero(x[:, i, :]) for i in range(x.shape[1])]
        axial_data.append(slices_pixel_data_temp)
        slices_pixel_data_temp = [np.count_nonzero(x[:, :, i]) for i in range(x.shape[2])]
        coronal_data.append(slices_pixel_data_temp)

    # gray to color transform
    tp = uf.np_3D_to_color(tp)
    fp = uf.np_3D_to_color(fp)
    fn = uf.np_3D_to_color(fn)
    color_roi = uf.gray_images_to_color(np.swapaxes(roi_images,0,2))

    #
    color_segmentation = np.copy(color_roi)

    color_segmentation[np.where((tp == [255, 255, 255]).all(axis=3))] = [0, 255, 0] # true positive in green
    color_segmentation[np.where((fp == [255, 255, 255]).all(axis=3))] = [255, 0, 0]  # false positive in red
    color_segmentation[np.where((fn == [255, 255, 255]).all(axis=3))] = [0, 0, 255]  # false negative in blue
    presentation_images =[]
    for i in range(np.shape(color_roi)[2]):
        img_a = np.swapaxes(np.array(color_roi)[:,:,i,:],0,1)
        img_b = np.swapaxes(color_segmentation[:,:,i,:],0,1)
        image = connect_2images(img_a, img_b)
        presentation_images.append(image)

    return presentation_images, pixelsdata, axial_data, coronal_data


# could

def connect_2images(img_a, img_b):

    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    total_width = wa + wb + 3


    max_height = np.max([ha, hb]) + 2

    new_img = np.ones(shape=(max_height, total_width, 3), dtype=np.uint8) * 255
    new_img[1:ha+1, 1:wa+1] = img_a
    new_img[1:hb+1, wa+2:wa+wb+2] = img_b
    #new_img[1:hc+1, wa+wb:wa+wb+wc] = img_c

    return new_img


def save_imgs(presentation, case, spacing, anim=True):
    fig = plt.figure()
    plts = []
    frames =[]
    directory = r"C:\Users\owner\Desktop\cases\roi_dicom\compare_seg#" + "(" + str(case) + ")"
    if not os.path.exists(directory):
        os.mkdir(directory)
    os.chdir(directory)
    num_of_slices = len(presentation)
    my_dpi = int(25.4 * (1/spacing))
    duration = 200
    begin = 0
    n,m,_ = presentation[1].shape
    plt.figure(figsize=(n / my_dpi, m / my_dpi), dpi=my_dpi)
    for i in range(begin,num_of_slices):
        image = presentation[i]

        plt.imshow(image)
        plt.axis('off')
        plt.savefig(str(i) + ".jpg")




