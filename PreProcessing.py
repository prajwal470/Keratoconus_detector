import cv2
import image_dehazer
import numpy as np
from matplotlib import pyplot as plt

lowest_y = []
upper_lst = []
lower_lst = []
mid_point = []
another_midpoint = []

def Denoiser(img):
    Denoised = cv2.fastNlMeansDenoisingColored(img, None, 4, 4, 7, 21)
    # cv2.imshow("Denoised", Denoised)
    # plt.imshow(Denoised)
    # plt.show()
    return Denoised


def Haze_removal(img):
    HazeCorImg = image_dehazer.remove_haze(img)
    HazeCorImg = cv2.cvtColor(HazeCorImg, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("HazeRemoved", HazeCorImg)
    # plt.imshow(HazeCorImg)
    # plt.show()
    return HazeCorImg


def HistEqualize(img):
    HistEquImg = cv2.equalizeHist(img)
    # cv2.imshow("HistogramEqualizedImage", HistEquImg)
    # plt.imshow(HistEquImg)
    # plt.show()
    return HistEquImg


def gradients(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    gradientmagnitude = cv2.magnitude(sobelx, sobely)

    # plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 2), plt.imshow(sobelx, cmap='gray')
    # plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 3), plt.imshow(sobely, cmap='gray')
    # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 4), plt.imshow(gradientmagnitude, cmap='gray')
    # plt.title('gradientmagnitude'), plt.xticks([]), plt.yticks([])
    # plt.show()
    return gradientmagnitude



def grabcut(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    # print(mask)
    # cv2.imshow('test', mask)
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)
    rectangle = (0, 10, 500, 350)
    # rect = cv2.rectangle(Original, (0,20), (500,350), (255,0,0), 2)
    # cv2.imshow('rect', rect)
    cv2.grabCut(image, mask, rectangle,
                backgroundModel, foregroundModel,
                3, cv2.GC_INIT_WITH_RECT)

    # In the new mask image, pixels will
    # be marked with four flags
    # four flags denote the background / foreground
    # mask is changed, all the 0 and 2 pixels
    # are converted to the background
    # mask is changed, all the 1 and 3 pixels
    # are now the part of the foreground
    # the return type is also mentioned,
    # this gives us the final mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # The final mask is multiplied with
    # the input image to give the segmented image.
    image = image * mask2[:, :, np.newaxis]

    plt.imshow(image)
    plt.colorbar()
    plt.show()
    return image


def contours(img):
    Draw_image = img
    Segmented = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(Segmented, 50, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(contours)

    cnt = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(Draw_image, cnt[0], -1, (0, 255, 0), 2)

    plt.imshow(Draw_image)
    plt.colorbar()
    plt.show()
    # cv2.imshow('Contours', Segmented)

    list_cnt = cnt[0].tolist()

    for i in list_cnt:
        lowest_y.append(i[0])
    lowest_y.sort()
    for j in range(len(lowest_y)):
        upper_lst.append(lowest_y[j][1])

    y_pls = sorted(upper_lst)
    for i in lowest_y:
        if i[1] == y_pls[0]:
            mid_point.append(i)

    temp = 0

    for i in mid_point:
        temp = temp + i[0]
    avg_1 = int(temp / len(mid_point))
    point_1 = [avg_1, y_pls[0]]

    # print(point_1)

    for i in range(len(lowest_y)):
        if (lowest_y[i][0] == avg_1) and (lowest_y[i][1] >= (y_pls[0] + 10)):
            lower_lst.append(lowest_y[i])
            point_2 = [lowest_y[i][0], lowest_y[i][1]]

            # CornialThickness = point_2[1] - point_1[1]

            # print(f"Cornial Thickness : {CornialThickness} pixels")


        elif (lowest_y[i][0] >= (avg_1 - 25)) and (lowest_y[i][0] <= (avg_1 + 25)) and (lowest_y[i][1] > (y_pls[0] + 10)):
            lower_lst.append(lowest_y[i])

    # print(lower_lst)

    temp = 0

    for i in range(len(lower_lst)):
        temp = temp + lower_lst[i][1]
    avg_2 = int(temp / len(lower_lst))
    point_2 = [avg_1, avg_2]
    # print(point_2)

    ba = np.array(lowest_y[0]) - np.array(point_1)
    bc = np.array(lowest_y[-1]) - np.array(point_1)

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    # print(np.degrees(angle))
    print(f"Curvature angle : {np.degrees(angle)} degrees")

    CornialThickness = avg_2 - y_pls[0]

    print(f"Cornial Thickness : {CornialThickness} pixels")

    return point_1, point_2


def Draw_points(img, lst):
    x_cor = lst[0]
    y_cor = lst[1]
    cv2.circle(img, (x_cor, y_cor), 4, (0, 255, 255), -1)


def features(path):
    OriginalImg = cv2.imread(path)
    Original = cv2.resize(OriginalImg, (500, 500))
    Input_Img = cv2.resize(OriginalImg, (500, 500))
    Denoised = Denoiser(Input_Img)
    HazeCorImg = Haze_removal(Denoised)
    HistEquImg = HistEqualize(HazeCorImg)
    GradImg = gradients(HistEquImg)
    Segmented = grabcut(Original)
    point_1, point_2 = contours(Segmented)
    Draw_points(Original, point_1)
    Draw_points(Original, point_2)
    plt.imshow(Original)
    plt.colorbar()
    plt.show()
    cv2.waitKey(0)