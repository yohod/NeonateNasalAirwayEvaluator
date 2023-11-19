import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import usefull_function as uf


# In slices that represent the entrance of air to the nose,
# the nostrils are open to the air surrounding the skull.
# Closing the nostrils is necessary to separate the air in the background of the skull from the air in the respiratory cavities.
# However, there is a trade-off between using a high threshold value to close the nostrils completely
# and using a low threshold value to avoid gaps between the nostrils and the respiratory tract.
# To address this issue, a two-step approach is proposed.
# In the first slice, the algorthm try to use the contour of the low threshold value of 700 HU
# to close the nostrils when it is possibile.
# In the second step, a high threshold value of HU 325 is used to close other slices.
# We will close the open nostrils with a straight line
# The line connects the end of the cartilaginous wing of the nose
# with the end of the cartilaginous septum of the nose on each side of the nostrils.

def closing_nostril(images, thresh, edge_index, end_open_nose_index):
    """
    Close open nostrils in a series of images. (main function)

    :param images: List of image slices.
    :param thresh: Threshold value used for binary image conversion.
    :param edge_index: Index marking the tip of the nose.
    :param end_open_nose_index: Index marking the end of the open nostril slices.

    :return: Modified images with closed nostrils and the index of the first open nostril slice.

    This function is designed to close open nostrils in a series of images where air enters the nose.
    It employs a two-step approach to handle the trade-off between closing nostrils completely
    and avoiding gaps between nostrils and the respiratory tract.
    In the first step, the algorithm attempts to use the contour of the low threshold value of 700 HU
    to close the nostrils when possible.
    In the second step, a higher threshold value of 325 HU is used to close other slices.
    The open nostrils are closed with straight lines connecting the end of the cartilaginous wing of the nose
    with the end of the cartilaginous septum of the nose on each side of the nostrils.
    The function iterates through the images,
    fixes the first open nostril image,
    prepares images for closing,
    finds points for closing the nostrils,
    and finally closes the nostrils with a straight line.
    """
    # 1. Fixing the first open nostril image with a lower threshold
    left_open_index, right_open_index = fixing_firsts_open(images, thresh, edge_index)
    open_nostril_index = min(left_open_index, right_open_index)
    open_nostril_index_copy = open_nostril_index

    # Images without open nostril images (like cases 16, 18, 26, 28)
    if open_nostril_index == end_open_nose_index:
        return images, open_nostril_index_copy

    # 2. Preparing images for closing - transforming to binary and cleaning
    bin_images = prepering_for_closing(images, thresh, open_nostril_index, end_open_nose_index)

    # 3. Find points and close the nostrils on the "real" image
    left_points = ((-1, -1), (-1, -1))
    right_points = ((-1, -1), (-1, -1))

    for image in bin_images[:6]:
        # Find the points for closing the nostrils
        left_points = find_left_point(image, left_open_index, open_nostril_index, left_points)
        right_points = find_right_point(image, right_open_index, open_nostril_index, right_points)
    # visualzation
        #contour, _ = cv.findContours(np.uint8(image), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        #contour = contour[0]
        # n,m = bin_images[0].shape
        # black_image = np.zeros((n,m,3))
        # cv.drawContours(black_image, contour, -1, (255,255,255))
        # ((x0,y0),(x1,y1)) = left_points
        # ((x2,y2),(x3,y3)) = right_points
        # black_image[x1,y1,:] = 255,0,0
        # black_image[x0,y0,:] = 0,0,255
        # black_image[x2,y2,:] = 255,0,0
        # black_image[x3,y3,:] = 0,0,255
        # print (left_open_index, right_open_index,open_nostril_index)
        #if False:
        #    print(open_nostril_index)
        #    image2 = np.stack((np.uint8(image),)*3, axis=-1)
        #    px0, py0 = left_points[0]
        #    px1, py1 = left_points[1]
        #    px2, py2 = right_points[0]
        #    px3, py3 = right_points[1]
        #    x_top, y_top =uf.top_pix(image)
        #    image2[x_top-1:x_top+1, y_top-1:y_top+1] = (255, 0, 0)
        #    image2[px1-1:px1+1, py1-1:py1+1] = (0,255,0)
        #    image2[px2-1:px2+1, py2-1:py2+1] = (0, 255, 0)
        #    image2[px0-1:px0+1, py0-1:py0+1] = (0,0,255)
        #    image2[px3-1:px3+1, py3-1:py3+1] = (0, 0, 255)
        #    plt.imshow(image2[:80,:])
        #    plt.show()

        # Closing the nostrils
        images[open_nostril_index] = close_contour(image, images[open_nostril_index], left_points, right_points)

        open_nostril_index += 1

    return images, open_nostril_index_copy



# find the first open nostril slice in each side.
# using the function "if_open" to check if side nostril is open

def first_open(bin_images, edge_index):
    """
    Find the first open nostril slice for each side in a series of binary images.

    :param bin_images: A list of binary images representing the nostrils.
    :param edge_index: Index marking the slice of the nose tip.

    :return: A tuple of integers indicating the number of the first open nostril slice for each side (left and right).

    This function processes a series of binary images to find the first slice where each nostril (left and right) is considered open. It iterates through the images and checks the status of each nostril, increasing the index when a nostril is closed. The process continues until both nostrils are open, and the function returns the slice numbers for the first open state of each nostril.

    If visualization is enabled (optional), the function may display intermediate images for debugging purposes.
    """
    n, m = bin_images[0].shape
    left_open_index = edge_index - 1
    right_open_index = edge_index - 1
    flag_left = False
    flag_right = False

    visul = False  # Visualization flag (optional)
    for image in bin_images[edge_index - 1:]:

        # Preprocessing the image
        image = uf.erase_object(image, 1000)  # Ignore images with disconnected objects
        x, y = uf.top_pix(image[:, m // 4:-m // 4])  # Exclude connected objects that are not the nose
        image[:x, :] = 0

        if not flag_left and not flag_right:
            flag_left, flag_right = if_open(image)
            if not flag_left:
                left_open_index += 1
            if not flag_right:
                right_open_index += 1
            if not flag_left and not flag_right and visul:
                contour, _ = cv.findContours(np.uint8(image), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                image2 = np.stack((np.uint8(image),) * 3, axis=-1)
                image2 = cv.drawContours(image2, contour, -1, (255, 0, 0), 1)
                n, m, _ = image2.shape
                image2[:, m // 10 - 2:m // 10] = (255, 128, 0)
                image2[:, 9 * m // 10:9 * m // 10 + 2] = (255, 128, 0)
                plt.imshow(image2[:70, :])
                plt.yticks([])
                plt.xticks([])
                plt.show()

        elif not flag_left:
            if visul:
                contour, _ = cv.findContours(np.uint8(image1), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                contour = contour[0]
                r_index = len(contour) // 2
                image2 = np.stack((np.uint8(image1),) * 3, axis=-1)
                image2[:, m // 10 - 2:m // 10] = (255, 128, 0)
                image2[:, 9 * m // 10:9 * m // 10 + 2] = (255, 128, 0)
                for i in range(r_index):
                    y, x = contour[i][0]
                    image2[x, y] = (255, 0, 0)

                for i in range(1, r_index):
                    y, x = y, x = contour[-i][0]
                    image2[x, y] = (0, 255, 0)
                plt.imshow(image2[:70, :])
                plt.xticks([])
                plt.yticks([])
                plt.show()
                visul = False
            flag_left, flag_right = if_open(image, side="left")
            if not flag_left:
                left_open_index += 1

        elif not flag_right:
            if visul:
                contour, _ = cv.findContours(np.uint8(image1), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                contour = contour[0]
                r_index = len(contour) // 2
                image2 = np.stack((np.uint8(image1),) * 3, axis=-1)
                image2[:, m // 10 - 2:m // 10] = (255, 128, 0)
                image2[:, 9 * m // 10:9 * m // 10 + 2] = (255, 128, 0)
                for i in range(r_index):
                    y, x = contour[i][0]
                    image2[x, y] = (0, 255, 0)
                for i in range(1, r_index):
                    y, x = y, x = contour[-i][0]
                    image2[x, y] = (255, 0, 0)
                plt.imshow(image2[:70, :])
                plt.xticks([])
                plt.yticks([])
                plt.show()
                visul = False
            flag_left, flag_right = if_open(image, side="right")
            if not flag_right:
                right_open_index += 1

        # When both nostrils are open, break the loop
        if flag_left and flag_right:
            visul = False
            if visul:
                contour, _ = cv.findContours(np.uint8(image), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                image2 = np.stack((image,) * 3, axis=-1)
                image2[:, m // 10 - 2:m // 10] = (255, 128, 0)
                image2[:, 9 * m // 10:9 * m // 10 + 2] = (255, 128, 0)
                image2 = cv.drawContours(image2, contour, -1, (0, 255, 0), 1)
                plt.imshow(image2[:70, :])
                plt.xticks([])
                plt.yticks([])
                plt.show()
            break
        image1 = np.copy(image)

    return left_open_index, right_open_index

def if_open(image, side="both"):
    """
    Check if the nostril is open in the given image.

    :param image: The image slice to analyze.
    :param side: Which side to check, either "both" (default), "left," or "right."

    :return: A tuple of Boolean values indicating the nostril status:
        - For the left nostril: True if open, False if closed.
        - For the right nostril: True if open, False if closed.

    The function analyzes the provided image to determine the open or closed status of the nostrils. It can focus on either both nostrils (the default) or a specific side (left or right).

    The analysis involves finding contours within the image and checking the direction of contour points to assess the openness of the nostrils.
    If the contour points move upwards or backwards in the image, it indicates open nostrils,
    and the respective flag is set to True.
    The function returns the status of both left and right nostrils as a tuple of Boolean values.
    """
    # Normalize which side to check
    r_flag = False
    l_flag = False
    if side == "left":
        r_flag = True
    elif side == "right":
        l_flag = True

    # Find contours
    contour, _ = cv.findContours(np.uint8(image), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = contour[0]

    # Contours go against clockwise, so for the right nostril, we need to begin from the end to start

    # Find if the left side is open
    # If the contour goes up more than 5 rows or goes back in the column without going down,
    # the nostril is open. We stop checking when the nostril closes until the column size/5.
    # x = row, y = column
    n, m = image.shape

    if not l_flag:
        up_counter = 0
        back_counter = 0
        y, x = contour[0][0]
        for cnt in contour[1:]:
            if y < m // 5:
                break

            if cnt[0][1] < x:
                y, x = cnt[0]
                up_counter += 1
                if up_counter >= 5:
                    l_flag = True
                    break

            if cnt[0][0] > y:
                y, x = cnt[0]
                back_counter += 1
                if back_counter >= 5:
                    l_flag = True
                    break
            else:
                if cnt[0][1] > x:
                    back_counter = 0
                y, x = cnt[0]

    if not r_flag:
        up_counter = 0
        back_counter = 0
        contour = contour[::-1]
        y, x = contour[0][0]
        for cnt in contour[1:]:
            if y > 4 * m // 5:
                break

            if cnt[0][1] < x:
                y, x = cnt[0]
                up_counter += 1
                if up_counter >= 5:
                    r_flag = True
                    break

            if cnt[0][0] < y:
                y, x = cnt[0]
                back_counter += 1
                if back_counter >= 5:
                    r_flag = True
                    break
            else:
                if cnt[0][1] > x:
                    back_counter = 0
                y, x = cnt[0]

    return (l_flag, r_flag)

# 1. trying to fix the firsts open nostril images using the contour of the lower threshold.
# sometimes in lower threshold the nostrils are still close
def fixing_firsts_open(images, thresh, edge_index):
    """
    Try to fix the first open nostril images using the contour of the lower threshold.

    :param images: A list of images representing the nostrils.
    :param thresh: Threshold value for binary images.
    :param edge_index: Index marking the beginning of the nostril slices.

    :return: A tuple of integers indicating the updated number of the first open nostril slice for each side (left and right).

    This function attempts to correct the first open nostril images in a series of images by using the contour information from lower threshold images. Sometimes, in lower threshold images, the nostrils may still appear closed. The function first identifies the first open slices for both left and right nostrils. If both nostrils are open, no corrections are made. If either or both nostrils are closed in the first open slices, the function draws the contour from lower threshold images onto the corresponding higher threshold images to simulate open nostrils.

    The function also updates the indices of the first open slices and removes the contour from the irrelevant side (if applicable). Visualization (optional) may be enabled to inspect the process.
    """
    # Find the first open nostril image for each side
    LOWER_THRESH = -300
    left_open_index, right_open_index = first_open(uf.binary_image(images, thresh), edge_index)
    index = min(left_open_index, right_open_index)
    if index == len(images):
        return left_open_index, right_open_index

    # Determine which side needs to be closed
    if left_open_index > right_open_index:
        side = "right"
    elif left_open_index < right_open_index:
        side = "left"
    else:
        side = "both"

    # Create a lower threshold image
    lower_thresh_img = np.uint8(uf.binary_image(images[index], LOWER_THRESH))
    lower_thresh_img = uf.erase_object(lower_thresh_img, 1000)

    # Check whether at least one nostril is closed in this image
    l_flag, r_flag = if_open(lower_thresh_img, side)
    if l_flag and r_flag:
        return left_open_index, right_open_index

    # If there is a closed nostril, draw the contour on the image to close the open nostrils in higher threshold
    contour, _ = cv.findContours(lower_thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    image = np.copy(images[index])
    n, m = image.shape
    cv.drawContours(images[index], contour, -1, (1500, 1500, 1500), 3)  #

    # Update the index of the open nostril and remove the contour from the irrelevant side
    if not l_flag:
        left_open_index += 1
    else:
        images[index][:50, :m // 2] = image[:50, :m // 2]
        images[index][:50, -m // 8:] = image[:50, -m // 8:]
    if not r_flag:
        right_open_index += 1
    else:
        images[index][:50, m // 2:] = image[:50, m // 2:]
        images[index][:50, :m // 8] = image[:50, :m // 8]

    # Visualization (optional)
    if False:
        # Display original image
        plt.imshow(image[:70], cmap=plt.cm.gray)
        plt.yticks([])
        plt.xticks([])
        plt.show()

        # Display binary image with a higher threshold
        image2 = uf.binary_image(image, 700)
        plt.imshow(image2[:70], cmap=plt.cm.gray)
        plt.yticks([])
        plt.xticks([])
        plt.show()

        # Overlay lower threshold contour on the higher threshold image
        image3 = np.uint8(lower_thresh_img) - np.uint8(image2)
        lower_thresh_img = np.uint8(np.stack((lower_thresh_img,)*3, axis=-1))
        lower_thresh_img[image3 == 255] = (255, 0, 0)
        plt.imshow(lower_thresh_img[:70])
        plt.yticks([])
        plt.xticks([])
        plt.show()

        # Display the updated image
        plt.imshow(images[index][:70], cmap=plt.cm.gray)
        plt.yticks([])
        plt.xticks([])
        plt.show()

    return left_open_index, right_open_index

# 2. preper binary images for the closing nostril operation
def prepering_for_closing(images, thresh, open_nostril_index, end_open_nose_index):
    """
    Prepare images for the nostril closing operation.

    :Parameters:
    images (list of 2D numpy arrays): A list of ROI images.
    thresh (float): Threshold value for binary images.
    open_nostril_index (int): Index marking the start of the open nostril.
    end_open_nose_index (int): Index marking the end of the open nostril.

    :Returns:
    list of 2D numpy arrays: Prepared binary images.

    :Details:
    - The function performs binary thresholding on the input images.
    - It finds the topmost position (x, y) in the images.
    - Adjusts the top position based on the thresholded image.
    - Iterates through the binary images and performs operations to prepare them for the nostril closing operation.
    - Provides optional visualization of the images before and after processing for debugging purposes.

    This function is responsible for preparing the binary images for the nostril closing operation.
    """
    # Reverse the open nostril slices to binary images
    bin_images = uf.binary_image(images[open_nostril_index:end_open_nose_index + 1], thresh)
    image = bin_images[0]
    n, m = image.shape

   # Find the top row (top_x) and the middle pixel (top_y) in the top row
    top_x, top_y = uf.top_pix(image[:, m // 2 - 7:m // 2 + 8])
    top_y = top_y + m // 2 - 7 + np.count_nonzero(image[top_x, m // 2 - 7:m // 2 + 8]) // 2
    index = 0
    # Erasing background objects (like pacifier and respiratory tube)
    for image in bin_images:
        image1 = np.copy(image)
        image = uf.erase_object(image, 5000)
        top_x += uf.top_pix(image[top_x:, top_y - 1:top_y + 2])[0]
        image[:top_x, :] = 0
        image[top_x, :top_y - 10] = 0
        index += 1

        # Visualization (optional)
        if np.count_nonzero(image1 - image) > 0 and False:
            print(open_nostril_index)
            plt.imshow(image1[:70], cmap=plt.cm.gray)
            plt.yticks([])
            plt.xticks([])
            plt.show()
            image3 = np.uint(image1 - image)
            image2 = np.stack((np.uint8(image1),)*3, axis=-1)
            image2[image3 > 0] = (255, 0, 0)
            plt.imshow(image2[:70])
            plt.yticks([])
            plt.xticks([])
            plt.show()

    return bin_images



# 3. find key points
def find_left_point(image, l_index, image_index, previous_points):
    """
     Find two key points on the top of the contours on the left side for closing the nostril with a straight line.

    :param image: The image slice.
    :param l_index: Index marking the open nostril slice on the left side.
    :param image_index: Index of the current image slice.
    :param previous_points: A tuple of previous points found for the left side.

    :return: A tuple of two points (x0, y0) and (x1, y1) to close the left nostril.

    This function identifies two key points for closing the left nostril with a straight line.
    The first key point is located at the end of the cartilage of the nasal septum.
    To find it, the search starts at the frontmost point of the nasal septum cartilage
    and iterates through the list of contour points, stopping when it finds a point followed by
    a change in direction towards the center of the image or the beginning of a sharp decline.
    The second key point is located at the frontmost row and is found by continuing to iterate
    through the contour list. If there is only a decline and no increase after it,
    the function chooses a point whose row is close to 2 rows (about 0.5 mm) to the center
    relative to the row of the second key point in the previous image.
    The function ensures that the points approach the center and takes into account the previous points' positions
    to determine the optimal points for closing the nostril.
    """

    (p0l, p1l) = previous_points
    # when the nostril close
    if l_index > image_index or image_index - l_index > 5:
        return ((-1, -1), (-1, -1))

     # Find the contour of the image
    contour, _ = cv.findContours(np.uint8(image), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = contour[0]



    # Initialize variables for the first key point (p1)
    y1, x1 = contour[0][0]
    contour_index = 1  # index of the point in the contour list

    # searching the first key point (p1)
    for cnt in contour[1:]:
        contour_index += 1
        # when the point go left
        if cnt[0][0] < y1:
            y1 = contour[contour_index - 1][0][0]  # = cnt[0][0]

        # when the point on the contour go right it's a sign for p1
        elif cnt[0][0] > y1:
            y1, x1 = contour[contour_index - 2][0]
            contour_index -= 1
            break

        # When the nose is shaped like a mushroom,
        # continue to the end of the head of the mushroom. Otherwise,
        # it is also a sign that we have reached the point.

        elif cnt[0][0] == y1 and l_index < image_index:
            ylist = [cnt[0][0] for cnt in contour[contour_index + 5:contour_index + 25]]
            if y1 + 5 in ylist:
                continue
            else:
                y1, x1 = contour[contour_index - 2][0]
                contour_index -= 1
                break

    # Ensure that the first key point (p1) approaches the center or starts a sharp decline
    if p1l[1] > y1 and l_index < image_index or y1 < 20:
        contour_index = 1
        x1 = contour[0][0][1]
        for cnt in contour[1:]:
            contour_index += 1
            if cnt[0][1] > x1:
                y1, x1 = contour[contour_index - 2][0]
                contour_index -= 1
                break


    # Initialize variables for the second key point (p0)
    x0, y0 = x1, y1

    # Continue searching for the second key point (p0)
    flag_min = False  # flag indicate a rise in the contour
    end_index = contour_index
    up_counter = 0
    for cnt in contour[contour_index:]:
        end_index += 1
       # If reached to the edge of the image
       # Then search for the point on the contour line that has the row value of p0 from the previous image
        if (y0 < 5 or y0 < p0l[1] - 10 or (p0l[1] < 20 and p0l[1] > 4 and y0 < p0l[1] and flag_min == False)
                and image_index > l_index):
            # print("no min")
            for cnt in contour[contour_index:end_index]:
                if cnt[0][0] == p0l[1]+2:
                    y0, x0 = cnt[0]
                    # print (p0l, (x0,y0))
                    break
            break
        # Restrict the key point to the location of the key point in the previous section image.
        # If it has not yet reached the end of the image, continue searching for an increase.
        if flag_min == False:
            if cnt[0][1] < x0:
                up_counter += 1
                if y0 < y1 - 1 and up_counter > 3:
                    same_row_counter = 0
                    flag_min = True
            y0, x0 = contour[end_index - 1][0]

        # if the rise begin, we search for the maximum point
        else:
            # continue to go up
            if cnt[0][1] < x0:
                same_row_counter = 0
                y0, x0 = contour[end_index - 1][0]
            # go down
            elif cnt[0][1] > x0:
                # There are exceptional cases in which there is a short-term decline and then an increase.
                # need to reach the maximum that is edge of the cartilage , If it returns to the height of the previous point
                xlist = [cnt[0][1] for cnt in contour[end_index:end_index + 10]]
                if x0 in xlist:
                    continue
                # If there is a right point and higher than the previous point, continue,
                # otherwise it is a decline that marks the key point

                else:
                    xlist = [cnt[0][1] for cnt in contour[end_index:end_index + 40]]
                    ylist = [cnt[0][0] for cnt in contour[end_index:end_index + 40]]
                    if max(ylist) > y0 and min(xlist) < x0:
                        continue
                    if same_row_counter > 1:
                        same_row_counter += 1
                    y0, x0 = contour[end_index - 2 - same_row_counter // 2][0]
                    break
            # if stay in the same row
            elif cnt[0][1] == x0:
                y0, x0 = contour[end_index - 1][0]
                same_row_counter += 1

    left_points = [(x0, y0), (x1, y1)]
    return left_points


def find_right_point(image, r_index, image_index, previous_points):
    """
    Find two key points on the top of the contours on the right side for closing the nostril with a straight line.

    :param image: The image slice.
    :param r_index: Index marking the open nostril slice on the right side.
    :param image_index: Index of the current image slice.
    :param previous_points: A tuple of previous points found for the right side.

    :return: A tuple of two points (x2, y2) and (x3, y3) to close the right nostril.

    This function identifies two key points for closing the right nostril with a straight line.
    The first key point is located at the end of the cartilage of the nasal septum.
    To find it, the search starts at the frontmost point of the nasal septum cartilage
    and iterates through the list of contour points, stopping when it finds a point followed by
    a change in direction towards the center of the image or the beginning of a sharp decline.
    The second key point is located at the frontmost row and is found by continuing to iterate
    through the contour list. If there is only a decline and no increase after it,
    the function chooses a point whose row is close to 2 rows (about 0.5 mm) to the center
    relative to the row of the second key point in the previous image.
    The function ensures that the points approach the center and takes into account the previous points' positions
    to determine the optimal points for closing the nostril.
    """

    (p2l, p3l) = previous_points
    if r_index > image_index or image_index - r_index > 5:
        return ((-1, -1), (-1, -1))

    n, m = image.shape
    contour, _ = cv.findContours(np.uint8(image), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = contour[0]

    # The contour's points list complete a counter-clockwise rotation. therefore,
    # reversing the contour list describes the right direction beginning in the highest point.
    contour = contour[::-1]

   # Initialize variables for the right first key point (p2)
    y2, x2 = contour[0][0]
    contour_index = 1

    # searching for the first key point
    for cnt in contour[1:]:
        contour_index += 1
        # When the direction of movement on the contour lines is right
        if cnt[0][0] > y2:
            y2 = contour[contour_index - 1][0][0]

        # If the direction of the point is left, it indicates that the key point has been found
        elif cnt[0][0] < y2:
            y2, x2 = contour[contour_index - 2][0]
            contour_index -= 1
            break

        elif cnt[0][0] == y2 and r_index < image_index:
            # When the nose is shaped like a mushroom,
            # continue to the end of the head of the mushroom.
            ylist = [cnt[0][0] for cnt in contour[contour_index + 5:contour_index + 25]]
            if y2 - 5 in ylist:
                continue
            else:
                y2, x2 = contour[contour_index - 2][0]
                contour_index -= 1
                break

    # Ensure that the first key point (p2) approaches the center or starts a sharp decline
    if p2l[1] < y2 and r_index < image_index or y2 > m - 20:
        contour_index = 1
        x2 = contour[0][0][1]
        for cnt in contour[1:]:
            contour_index += 1
            if cnt[0][1] > x2:
                y2, x2 = contour[contour_index - 2][0]
                contour_index -= 1
                break


   # Initialize variables for the second key point (p0)
    x3, y3 = x2, y2

    # Continue searching for the second key point (p3)
    same_row_counter = 0
    flag_min = False  # indicate a rise in the contour
    end_index = contour_index
    up_counter = 0
    for cnt in contour[contour_index:]:
        end_index += 1

        # If reached to the edge of the image
        # Then search for the point on the contour line that has the row value of p3
        # from the previous image

        if (y3 > m - 6 or (y3 > p3l[1] + 15 or p3l[1] > m - 20 and p3l[1] < m - 7 and
                           y3 > p3l[1] and flag_min == False) and r_index < image_index):
            # print("no min")
            for cnt in contour[contour_index:end_index]:
                if cnt[0][0] == p3l[1]-2:
                    y3, x3 = cnt[0]
                    break
            break

        if flag_min == False:
            if cnt[0][1] < x3:
                up_counter += 1
                if y3 > y2 + 1 and up_counter > 3:
                    flag_min = True
            y3, x3 = contour[end_index - 1][0]
        else:
            if cnt[0][1] < x3:
                same_row_counter = 0
                y3, x3 = contour[end_index - 1][0]
            elif cnt[0][1] > x3:
                xlist = [cnt[0][1] for cnt in contour[end_index:end_index + 10]]
                if x3 in xlist:
                    continue
                else:
                    ylist = [cnt[0][0] for cnt in contour[end_index:end_index + 40]]
                    xlist = [cnt[0][1] for cnt in contour[end_index:end_index + 40]]
                    if min(ylist) < y3 and min(xlist) < x3:
                        continue
                    if same_row_counter > 1:
                        same_row_counter += 1
                    y3, x3 = contour[end_index - 2 - same_row_counter// 2][0]
                    break
            elif cnt[0][1] == x3:
                y3, x3 = contour[end_index - 1][0]
                same_row_counter += 1

    right_points = [(x2, y2), (x3, y3)]
    return right_points


# 4.closing nostril
def close_contour(bin_image, image, left_points=((-1, -1), (-1, -1)), right_points=((-1, -1), (-1, -1))):
    """
    Close the nostrils in the real image using binary information and key points.

    :param bin_image: Binary image for closing the nostrils.
    :param image: The original image to close the nostrils in.
    :param left_points: A tuple of two key points for closing the left nostril.
    :param right_points: A tuple of two key points for closing the right nostril.

    :return: The original image with the nostrils closed.

    This function takes a binary image with marked key points
    for closing the nostrils and the original image.
    It closes the nostrils by drawing straight lines in the binary image
    based on the provided key points for the left and right nostrils.
    Then, it extracts the contour of the closed nostril in the binary image
    and creates a mask where the air outside the closed nostrils is relatively white (1000 HU).
    Finally, the function applies the mask to the original image,
    effectively closing the nostrils in the real images by modifying the pixel values
    where the nostrils were previously open.
    This method was designed to close the nostrils even in high HU value scenarios,
    allowing flexibility in choosing the threshold value during the segmentation step.
    """

    # Close the left and right nostrils with a straight line in the binary image
    if left_points[0][1] != -1:
        cv.line(bin_image, (left_points[0][1], left_points[0][0]), (left_points[1][1], left_points[1][0]),
                (2 ** 16 - 1, 2 ** 16 - 1, 2 ** 16 - 1))
    if right_points[0][1] != -1:
        cv.line(bin_image, (right_points[0][1], right_points[0][0]), (right_points[1][1], right_points[1][0]),
                (2 ** 16 - 1, 2 ** 16 - 1, 2 ** 16 - 1))

    # Find the contour of the closed nostril in the binary image
    contour, _ = cv.findContours(np.uint8(bin_image), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Create a mask where all the air outside the closed nostrils is relatively white (1000 HU)
    black_image = np.zeros_like(image)
    cv.drawContours(black_image, contour, -1, (1000, 1000, 1000), -1)
    black_image = 1000 - black_image

    # Close the nostrils in the original image using the mask image
    image = image + black_image

    return image
