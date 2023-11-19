import numpy as np
import os
import usefull_function as uf
from PIL import Image
import concurrent.futures



def presentation(slices, roi_images, seg_images, fix_seg_images, plane_mode = "coronal", additional_slices =[],
                 reverse=False, present_improvment=False):
    """
    color the segmented airway area. in red the connected area, in orange disconnected area.
    when present_improvment is True color the first segmented area in red and the fix in yellow
    without considering the difference between a connected area and disconnected
    :param slices: ct slices. only the slices that include airway
    :param roi_all_slices: all slices, roi images
    :param roi_images: roi slices, and roi images
    :param seg_images: segmented airway images
    :param fix_seg_images: segmented airway after using adaptive thresholding
    :param reverse: TRUE if the ct images were reversed
    :param method:
    """


    presentation_images = []
    color_roi = uf.gray_to_color(roi_images)
    color_segmentation = np.copy(color_roi)


    # showing only the fix segmentation
    if present_improvment is False:
        # take only the united big part of the segmentation images
        fix_airway_images = uf.remove_unconnected_objects(fix_seg_images)
        # gray to color transform
        color_fix_airway_images = uf.np_3D_to_color(fix_airway_images)
        # mapping the disconnected regions
        color_erase_fix_images = uf.np_3D_to_color(fix_seg_images) - color_fix_airway_images

        # coloring the connected region in red and the disconnected in orange
        color_segmentation[np.where((color_fix_airway_images == [255, 255, 255]).all(axis=3))] = [255, 0, 0]
        color_segmentation[np.where((color_erase_fix_images == [255, 255, 255]).all(axis=3))] = [255, 128, 0]

    # if wanted to present the differences between the segmentation before using the local threshold and after
    else:# present_improvment is True
        color_fix_airway_images = uf.np_3D_to_color(fix_seg_images)
        color_airway_images = uf.np_3D_to_color(seg_images)
        # mapping the differences
        color_differences = color_fix_airway_images  - color_airway_images

        # coloring the first segmentation in red and the improvement in yellow
        color_segmentation[np.where((color_differences == [255, 255, 255]).all(axis=3))] = [255, 255, 0]
        color_segmentation[np.where((color_airway_images == [255, 255, 255]).all(axis=3))] = [255, 0, 0]



    if plane_mode == "sagittal" and len(additional_slices) > 0:

    # Marking the selected axial slices
        if reverse == False:
            begin = slices[0].InstanceNumber - 1
            end = slices[-1].InstanceNumber - 1
        else:
            begin = - slices[0].InstanceNumber
            end = - slices[-1].InstanceNumber
        color_additional = uf.gray_images_to_color(additional_slices, slope, intercept)
        color_additional[begin, :] = (255, 0, 0)
        color_additional[end, :] = (255, 0, 0)
        # resize the images to be 3 times bigger. and swap the image axis to produce sagittal images
        color_additional = uf.axial_to_sagittal(resize3d(color_additional))
        color_roi = uf.axial_to_sagittal(resize3d(color_roi))
        color_segmentation = uf.axial_to_sagittal(resize3d(color_segmentation))

    elif plane_mode == "coronal":
        # resize the images to be 3 times bigger. and swap the image axis to produce coronal images
        color_roi = uf.axial_to_coronal(resize3d(color_roi))
        color_segmentation = uf.axial_to_coronal(resize3d(color_segmentation))

    elif plane_mode == "axial" and len(additional_slices) > 0:
        color_additional = uf.gray_to_color(additional_slices)

    for i in range(np.shape(color_roi)[0]):
        if plane_mode == "coronal" or len(additional_slices) == 0:
            new_image = connect_images((color_roi[i], color_segmentation[i]))
        else:
            new_image = connect_images((color_additional[i], color_roi[i], color_segmentation[i]))
        presentation_images.append(new_image)

    return presentation_images

def resize(image):
    # multiply the image height 3 times for a gray image
    n, m, f = image.shape
    fixed_image = np.empty(shape=(3 * n, m, f), dtype=np.uint8)
    for i in range(n):
        fixed_image[3 * i:3 * i + 3] = image[i]
    return fixed_image

def resize3d(images):
    # muliply the image height 3 times for an RGB image
    n, m, f,_ = np.shape(images)
    fixed_image = np.empty(shape=(3 * n, m, f,3), dtype=np.uint8)
    for i in range(n):
        fixed_image[3 * i:3 * i + 3] = images[i]
    return fixed_image



def connect_images(tuple_of_images):
    "connected some images with a white column different them"
    height = 0   # height for the connected image
    width = 1   # width for the connected image
    for image in tuple_of_images:
        n,m,p = image.shape
        if n > height:
            height = n + 2
        width += m + 1
    new_img = np.ones(shape=(height,width,p), dtype=np.uint8) * 255
    width = 1
    for image in tuple_of_images:
        n, m, p = image.shape
        new_img[1:n + 1, width:width+m,:] = image
        width += m + 1
    return new_img



def save_img(presentation, save_path=None, mode='axial', anim=True):
    if save_path is None:
        save_path = uf.select_location("select folder for saving the images")
    directory = save_path + "\\" + mode + " segmentation images"
    if not os.path.exists(directory):
        os.mkdir(directory)
    os.chdir(directory)

    num_of_slices = len(presentation)
    duration = 200
    begin = 0

    if mode == "coronal":
        num_of_slices -= 10
        begin = 3
        duration = 100

    # Save images in parallel using ThreadPoolExecutor to shorten the image saving time
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        frames = []
        for i in range(begin, num_of_slices):
            image = presentation[i]
            output_path = f'{i}.jpg'
            futures.append(executor.submit(save_image, image, output_path))
            frames.append(Image.fromarray(image))

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

    # create an animation
    if anim is True:
        frames[0].save(
           save_path + mode + '.gif',
            save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)


# Function to save an image
def save_image(image_array, output_path):
    image = Image.fromarray(image_array)
    image.save(output_path, format='JPEG', quality=90)


