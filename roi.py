import numpy as np
import cv2 as cv
import usefull_function as uf

# 2.1 cutting images side. first roi cutting
def cutting_sides(images, images_for_sagittal):
    """ cutting  the images sides according to the head sides,
    and taking the 2/4 middle between head sides. the nasal airway are in the middle of the head"""
    ROI_THERSH = -525
    length,n, m = images.shape

    # taking the middle axial image of the ct scan. (A slice where the skull surely is not too small)
    # Binary image. Bones and tissues in white.
    # remove noise objects
    image = images[length // 2]
    bin_img = uf.binary_image(image, ROI_THERSH)
    bin_img = uf.erase_object(bin_img, 10000)

    # Searching the skull by a significant number of white pixels in a column
    # (which represent tissue and bones)
    # find the left side of the head
    y_left = 0
    for i in range(m - 1):
        count_white = np.count_nonzero(bin_img[:, i])
        if count_white < n / 4:
            y_left += 1
        else:
            break

    # find the right side of the head
    y_right = -1
    for i in range(-2, -m, -1):
        count_white = np.count_nonzero(bin_img[:, i])
        if count_white < n / 4:
            y_right -= 1
        else:
            break

    # the nose is in the middle of the head
    # take the 2 middle quarter of the skull
    width = m + y_right - y_left
    width_factor = width // 4
    y_left = y_left + width_factor
    y_right = y_right - width_factor
    images = images[:,:, y_left:y_right]
    images_for_sagittal = images_for_sagittal[:,:, y_left:y_right]
    return images, images_for_sagittal


# 2.2 finding the first nasal airway slice
# take 5 slices just in case of low threshold
# the first air object in neonate head normally is the nasal cavity
# the frontal sinus begins to develop usually at 12 months [https://doi.org/10.1016/j.otot.2018.03.002]
def find_first_airway(bin_images):
    """ receive the binary images of the head. and find the first slice which the nasal_airway
    """
    num_element = img_element(bin_images, row_factor=0.5)

    start = 0
    # find the first image with
    while (num_element[start]) == 0 or list_in([0] * 10, num_element[start:-50]):
        start += 1

    # taking into account cases where the threshold value at which we converted the images to binary
    # was small. and add some axial slices (about 2.5-5 mm)
    num_of_slices = 5
    start -= num_of_slices
    if start < 0:
        start = 0

    return start


# helping functions for "find_first_airway"

# 2.2.1 count air objects in each image in the list of head images
def img_element(images, row_factor=1.0, erase_factor=0):
    """
    Count the number of air objects in each image. The head and air around the head are not counted
    :param row_factor: this factor manipulate the part of the image we want to count the objects.
    need to be between 0-1. default  is 1
    :param erase_factor: this factor indicate the min size of an object to be counted.  default is 0
    :type  images:  3D numpy array.
    :return: a list of the number of air objects in each image.
    :rtype:list of ints
    """
    num_el = []
    n, m = images[0].shape
    x_end = int(n * row_factor)
    for image in images:
        # counting the
        inv_image = 255 - np.uint8(image[:x_end, :])
        if erase_factor != 0:
            inv_image = uf.erase_object(inv_image, erase_factor)
        num_el.append(uf.cc_stat(inv_image)[0] - 2) # minus 2 is for discount the air in the background and the head object
    return num_el


# 2.2.2
# check if list 1 is in list 2.
# help to find if there are lot of slices in the continue of the image list with 2 objects
# this help to find the real first object with nasal airway
def list_in(l1, l2):
    for i in range(len(l2) - len(l1)):
        if l1 == l2[i:i + len(l1)]:
            return True
    return False


# 2.3 find the image of the nose tip
# the nose tip is the most anterior pixels of the nose
# Searching for the slice where the nose pixels are most anterior
def find_edge_image(bin_images):
    """ find the image of the nose edge. by searching a maximum in top pixel of the image
    :return:edge index and the top row pixel"""

    x_top = 400
    edge_index = 19 - 1
    # the image of the edge wouldn't be in the begging of the nasal airway.
    # need to jump on begging because sometimes top_pixel in the forehead are topper then in the edge of the nose.
    # breaking the loop when the nose become "smaller"
    # find the image with the anterior pixel of the nose
    for image in bin_images[19:]:
        image = uf.erase_object(image, 10000)  # clean images
        x = uf.top_pix(image)[0]
        if x <= x_top:
            x_top = x
            edge_index += 1
        else:
            break
    return edge_index


# 2.4 cutting images sides and top (roi)
def cutting_roi_top_and_sides(images, bin_images, images_for_sagittal, edge_index, old):
    """cutting the top of the images accorrding to top pixel of the edge image and
    " cutting sides of the images according to the center pixel of the nose tip,
     taking about the middle third of the head, the sinuses have not yet developed
     (In the case of an adult, this part should be removed)
      """


    image = bin_images[edge_index]

    x, y_top = uf.top_pix(image)
    y_top += np.count_nonzero(image[x, :]) // 2
    n, m = image.shape


    for row in range(x, x + 100):
        count_white = np.count_nonzero(image[row, :])
        if count_white > m / 2:
            break

    # cutting the image. good for neonate, not for bigger ages (that have sinus)
    if not old:
        #setting the cut parameter (in all with the first cut, it take the middle third of the head)
        cut_factor = int(m * 0.35)
        y_left = y_top - cut_factor
        y_right = y_top + cut_factor
        if y_left < 0:
            y_left = 0
        elif y_right > m:
            y_right = m
        images = images[:, x:, y_left:y_right]
        bin_images = bin_images[:,x:, y_left:y_right]
        images_for_sagittal = images_for_sagittal[:,:, y_left:y_right]

    else:
        images = images[:,x:, :]
        bin_images = bin_images[:,x:, :]
    return images, bin_images, images_for_sagittal


# 2.5 find the end slices of the open nostrils
# when the  anterior pixel of the nose is minimal, this means that we below the nostril slices
# should be improved to detect more precisely the end slice

def find_end_nostril_slices(bin_images, edge_index):
    """find the last image of the open nostrils
    :param edge_index: this is the index of the edge of the image """


    interferent = False  #

    # the nostril is At least 5 slices below the tip of the nose.
    if edge_index + 6 < len(bin_images):
        stop_index = edge_index + 6
        image = bin_images[edge_index + 5]
        image = uf.erase_object(image, 3000)
        x_minimom, y = uf.top_pix(image)

    # search for minimum on the top pixel
        for image in bin_images[edge_index + 6:]:
            image = uf.erase_object(image, 3000)  # מסיר אובייקטים נפרדים מהגולוגולת
            x = uf.top_pix(image)[0]
            if x > x_minimom:
                x_minimom = x
                stop_index += 1
            else:
                break
        num = uf.cc_stat((255 - image[:35, :]))[0]
        if num > 0:
            interferent = True
            stop_index += 5  # if there is an interfernt, take additional slices just to be sure

    else:
        stop_index = len(bin_images)
    return stop_index, interferent


# 2.6. approximate the last slice of the nasal airway. including the nasopharynx
# before it is connecting the oral cavities (the area of the nasopharynx growing smaller
# when connecting the oral cavities the area increase
# also cut the posterior part of the head (behind the nasopharynx)
def find_end_nasopharynx(images, bin_images, edge_index, end_open_nose_index):
    """ approximate the last slice of the nasal airway.
    distinguishes two cases: when the ct images was quasi perpendicular(image_type=1)
    to the head or the head is tilted forward (image_type=2). if the airway continue after the open nostril it mean is not perpendicular.
    # from an axial nasopharynx slice search when the nasopharynx have minimum area.
    # because in the oropharynx the area become bigger with the connection with other objects under the soft palate
    :return : stop slice index, and image_type.
    """

     # if the interior nasal airway continue after the open nostril it mean is surly not perpendicular

    index = 0
    num_element = img_element(bin_images[edge_index:], row_factor=0.4)
    while (len(num_element)) > index and (num_element[index] > 3):
        index += 1
    surly_airway_index = index + edge_index

    # not perpendicular - the head is tilted forward
    if surly_airway_index > end_open_nose_index:
        image_type = 2
        surly_naso_index = surly_airway_index + 15 # adding 15 slices to be in the nasopharynx slices
        end_nasopharynax, down_row = find_min_pharynx(bin_images, surly_naso_index)

    # quasi perpendicular to the head
    else:
        image_type = 1
        surly_naso_index = find_widest_object(bin_images, edge_index, end_open_nose_index)
        end_nasopharynax, down_row = find_min_pharynx(bin_images, surly_naso_index)

    # cutting the bottom of the images under the nasopharynx
    images = images[:,:down_row,:]
    return images, surly_naso_index, end_nasopharynax, image_type


# helping function for "find_end_nasopharynx"
# find the widest object
# want to find the nasopharynx. it is the widest object. (sometimes the mouth is wider so search from 0.3n of the rows
# or the orophrynax so take care and not go to the end of the open nose_index if we found a wide enough object before

def find_widest_object(images, edge_index, end_open_nose_index):

    start_index = edge_index - 15
    stop_index = end_open_nose_index+5
    index = start_index
    max_wid = 0
    max_ind = 0
    n, m = images[0].shape

    # searching an axial slice where the airway is connected to the nasopharynx (wide _object)
    # the nasopharynx is in the back of the head

    for image in images[start_index:stop_index]:
        image = 255 - image[3 * n // 10:n // 2,:]
        num, label, stat, cent = uf.cc_stat(image)
        if num > 1:
            width = np.amax(stat[1:, cv.CC_STAT_WIDTH])
        else:
            width = 0
        if width >= max_wid:
            max_wid = width
            max_ind = index
        if max_wid > 15 and index >= end_open_nose_index - 5:
            break
        index += 1
    return max_ind

# find the axial slice where the pharnax area is minimal

def find_min_pharynx(images, begin):

    n, m = images[0].shape
    down = -1
    min_pharynx = 0 # minimum area nasopharynx
    min_pharynx_index = begin
    image = images[begin]

    # remove the air in the back of the head by find a row of white pixels (head)
    for down in range(down, -n // 2, -1):
        nonzero = np.count_nonzero(image[down, :])
        if nonzero == m:
            break

    first_top = (n + down) // 4 # remove the part that is more anterior than the nasopharnx ,
    top = first_top
    decrease_flag = False # when the area begin to decrease
    down_pharynx = -1

    for image in images[begin:]:

        # remove the air in the back of the head by find a row of white pixels (head)
        for down in range(down, -n // 2, -1):
            nonzero = np.count_nonzero(image[down, :])
            if nonzero == m:
                break
        # inverse the images for find the airway object data
        image = 255 - image[top:down, :]
        image = uf.erase_object(image, 10)
        num, label, stat, cent = uf.cc_stat(image)

        # search for the nasopharynx which is more posterior than the mouth and relatively large
        if num > 2:
            # find the 2 biggest objects (nasopharynx and perhaps the mouse)
            first_biggest_arg = np.argmax(stat[1:, cv.CC_STAT_AREA]) + 1
            second_biggest_arg = np.argsort(np.array(stat[1:, cv.CC_STAT_AREA]))[-2] + 1

            # compare which one is more posterior in the image
            if stat[first_biggest_arg, cv.CC_STAT_TOP] < stat[second_biggest_arg, cv.CC_STAT_TOP] \
                    and stat[second_biggest_arg, cv.CC_STAT_AREA] > 400:
                pharynx_area = stat[second_biggest_arg, cv.CC_STAT_AREA]
                down_pharynx = top + stat[second_biggest_arg, cv.CC_STAT_TOP] + stat[second_biggest_arg, cv.CC_STAT_HEIGHT]
                # remove the part that is more anterior than the nasopharnx
                #To avoid a mistake in identifying the pharynx in the next section
                top = max(stat[second_biggest_arg, cv.CC_STAT_TOP] - 10 + top, first_top)
            else:
                pharynx_area = stat[first_biggest_arg, cv.CC_STAT_AREA]
                down_pharynx = top + stat[first_biggest_arg, cv.CC_STAT_TOP] + stat[first_biggest_arg, cv.CC_STAT_HEIGHT]
                # remove the part that is more anterior than the nasopharnx
                # To avoid a mistake in identifying the pharynx in the next section
                top = max(stat[first_biggest_arg, cv.CC_STAT_TOP] - 10 + top, first_top)

        # when there is only 2 objects, the first one is the head tissues and the additonal object is surly the nasopharynx
        elif num == 2:
            pharynx_area = stat[1, cv.CC_STAT_AREA]
            down_pharynx = top + stat[1, cv.CC_STAT_TOP] + stat[1, cv.CC_STAT_HEIGHT]
            # remove the part that is more anterior than the nasopharnx
            # To avoid a mistake in identifying the pharynx in the next section
            top = max(stat[1, cv.CC_STAT_TOP] - 10 + top, first_top)

        # where tha location of the nasopharynx is the axial slice is still not in the 0.3 rows of the images,
        # need to continue
        else:
            continue

        # 3 cases:
        # 1. became bigger, continue until it became smaller
        # 2. became smaller than flag=True and continue.
        # 3. stop when pharynx have min area.

        if pharynx_area <= min_pharynx:
            min_pharynx = pharynx_area
            if min_pharynx_index - begin > 4:
                decrease_flag = True

        elif decrease_flag == False:
            min_pharynx = pharynx_area

        elif decrease_flag == True:
            break

        min_pharynx_index += 1

    down_pharynx = min(down_pharynx+30, n+down) # the back location where to cut the image

    return min_pharynx_index, down_pharynx



# 2.7
# The main function that performs all the steps for select the Volume Of Interest (VOI)
def voi(slices, images_for_sagittal, images, old=False):
    """ take of the images only the Region Of Interest slices and region (nose and nasopharynx)
    images_for_sagittal: images for sagttial presentation
    :return
    # slices - nose slices
    # all_head - nose slices, all head images after preprocessing
    # roi_all_slices - cropped sides images for all slices for the segittal presnation
    # images - roi of nasal airway images
    # edge_index - index of the edge of the nose
    # end_open_nose_index - index of the last nostril image
    # naso_index
    # end_nasopharynax - the index of the image of the end of nasopharynx
    # image_type:  1 - normal slices. 2- The head is tilted backword. 3. The head is tilted forward
    # interferent - True when there is an interferent in the nostril
    """
    thresh = -525
    # images for axial presentation. just with preprocessing changing
    all_head = np.copy(images)

    # 1. first cutting
    images, images_for_sagittal = cutting_sides(images, images_for_sagittal)
    bin_images = uf.binary_image(images, thresh)

    # 2. find the first airway image + 5 slices before
    start = find_first_airway(bin_images)

    # chosing the slices which begging in tha nasal airway
    bin_images = bin_images[start:]
    images = images[start:]
    slices = slices[start:]
    all_head = all_head[start:]

    # 3. find nose edge image
    edge_index = find_edge_image(bin_images)

    # 4.cutting sides and top
    images, bin_images, images_for_sagittal = cutting_roi_top_and_sides(images, bin_images, images_for_sagittal,
                                                                        edge_index, old)

    # 5. find the end_slices of the open-nostril
    end_open_nose_index, interferent = find_end_nostril_slices(bin_images, edge_index)

    # 6. approximate the last nasopharynx slice and cutting the bottom of the images
    images,naso_index, end_nasopharynax, image_type = find_end_nasopharynx(images, bin_images, edge_index, end_open_nose_index)
    stop = max(end_nasopharynax, end_open_nose_index)

    # The head is tilted forward ?
    # when the last axial slice of the open nostril is inferior from the last axial slice of the nasopharynx
    if end_nasopharynax < end_open_nose_index:
        image_type = 3

    # taking only the relevant slices
    slices = slices[:stop]
    images = images[:stop]
    all_head = all_head[:stop]

    return slices, all_head, images_for_sagittal, images, edge_index, end_open_nose_index, naso_index, end_nasopharynax, image_type, interferent


