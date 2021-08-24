import cv2
import numpy as np
import math
import imutils

COLOR_MIN = np.array([0, 15, 10], np.uint8)
COLOR_MAX = np.array([7, 255, 255], np.uint8)


def empty(a):
    pass


def my_hough_lines(canny_frame):
    cv2.imshow("canny_frame",canny_frame)
    w, h = canny_frame.shape
    r_max = int(np.sqrt(w ** 2 + h ** 2))
    r_list = np.arange(-r_max, r_max)
    theta_list = np.arange(np.pi, step=np.pi / 180)
    # Hough accumulator array of theta vs rho
    hough_space = np.zeros((len(r_list), len(theta_list)), dtype=np.uint64)  # accumulator matrix
    yy, xx = np.nonzero(canny_frame)  # (row, colo) indexes to edges

    # Vote in the Hough accumulator
    t = np.zeros(len(theta_list))
    r = np.zeros(len(r_list))
    for x, y in zip(xx, yy):
        for t_idx, theta in enumerate(theta_list):
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = int(round(rho)) + r_max
            hough_space[rho_idx, t_idx] += 1
            t[t_idx] = theta
            r[rho_idx] = rho
    return hough_space, r_max, r, t


def open_parameters_config_window():
    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", (640, 120))
    cv2.createTrackbar("Threshold1", "Parameters", 50, 255, empty)
    cv2.createTrackbar("Threshold2", "Parameters", 150, 255, empty)
    cv2.createTrackbar("Area", "Parameters", 1300, 15000, empty)


def draw_line(frame, theta, rho):
    if not isinstance(frame, np.ndarray):
        return None
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    cv2.line(frame, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)


def is_line_vertical(is_goal_horizontal, theta):
    # if the degree of the line is vertical return true
    return np.logical_and(not is_goal_horizontal, np.logical_or(0.4 * np.pi <= theta <= 0.6 * np.pi,
                                                                1.4 * np.pi <= theta <= 1.6 * np.pi))


def is_line_horizontal(is_goal_horizontal, theta):
    # if the degree of the line is horizontal return true (4 cases)
    return np.logical_or(
        np.logical_and(is_goal_horizontal, np.logical_or(0 * np.pi <= theta <= 0.1 * np.pi,
                                                         1.9 * np.pi <= theta <= 2 * np.pi)),
        np.logical_and(is_goal_horizontal, np.logical_or(0.9 * np.pi <= theta <= 1 * np.pi,
                                                         1 * np.pi <= theta <= 1.1 * np.pi))
    )


def draw_goal_lines(r, t, frame, is_goal_horizontal):
    if not isinstance(frame, np.ndarray):
        return None

    if r is not None:

        if is_line_horizontal(is_goal_horizontal, t):
            draw_line(frame, t, r)

        elif is_line_vertical(is_goal_horizontal, t):
            draw_line(frame, t, r)

        else:
            empty(0)


def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        print(curr_frequency)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


def merge_lines(r, t, left_to_right):
    if np.logical_and(len(r) == 1, len(t) == 1):
        return r, t

    most_common_t = most_frequent(t)

    for i, value in enumerate(t):
        if value != most_common_t: #somthing went worng here because there is values in t that not equal to most_coomon_t
            print("value ={} =!".format(value), "most_common={}".format(most_common_t))
            t.remove(value)
            r.remove(r[i])
        else:
            print("value ={} =".format(value), "most_common={}".format(most_common_t))

    maximum = max(r)
    minimum = min(r)

    print("theta = {}".format(t), "r = {}".format(r))
    if left_to_right:
        new_r = maximum
    else:
        new_r = minimum

    print("new_r = {}".format(new_r), "new_t = {}".format(most_common_t))
    return new_r, most_common_t


def get_lines_from_accumulator(H, r_max, r_array, t_array, threshold_for_lines):
    yy, xx = np.nonzero(H)
    r = []
    t = []

    for x, y in zip(xx, yy):
        if H[y, x] >= threshold_for_lines:
            r.append(r_array[y])
            t.append(t_array[x])
    rho, theta = merge_lines(r, t, left_to_right=True)
    return rho, theta


def get_goal_lines(frame, frame_canny, is_goal_horizontal, threshold_for_lines):
    if not isinstance(frame, np.ndarray):
        return None
    if not isinstance(frame_canny, np.ndarray):
        return None

    # get lines
    # lines = cv2.HoughLines(frame_canny, 1, np.pi / 180, threshold_for_lines, None, 0, 0)

    H, r_max, r_array, t_array = my_hough_lines(frame_canny)

    return get_lines_from_accumulator(H, r_max, r_array, t_array, threshold_for_lines)


# def draw_ball_contours(contour, frame, frame_contours):
#     if not isinstance(frame, np.ndarray):
#         return None
#     if not isinstance(frame_contours, np.ndarray):
#         return None
#
#     if cv2.isContourConvex(contour):
#         frame_contours = cv2.convexHull(contour)
#     else:
#         (x, y), radius = cv2.minEnclosingCircle(contour)
#         if np.pi * (radius ** 2) < 0.05 * frame.shape[0] * frame.shape[1]:
#             center = (int(x), int(y))
#             radius = int(radius)
#             frame_contours = cv2.circle(frame_contours, center, radius, (0, 255, 0), 2)


def get_center_and_radius(mask):
    if not isinstance(mask, np.ndarray):
        return None
    # get contours
    # contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # draw contours
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     area_from_user = cv2.getTrackbarPos("Area", "Parameters")
    #     if area > area_from_user:
    #         draw_ball_contours(contour, frame, frame_contours)
    # return area
    center = None
    radius = 0
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return center, radius


def draw_ball(center, radius, result):
    cv2.circle(result, center, int(radius), (0, 255, 0), 2)


# def draw_ball_hough_circle(x, y, r, frame, result):
#     if not isinstance(frame, np.ndarray):
#         return None
#     area_from_user = cv2.getTrackbarPos("Area", "Parameters")
#     area = np.pi * (r ** 2)
#     if area_from_user < area < (0.06 * frame.shape[0] * frame.shape[1]):
#         cv2.circle(result, (x, y), r, (255, 0, 0), 3)
#
#     return area


# def get_ball_hough_circles(frame, result):
#     if not isinstance(frame, np.ndarray):
#         return None
#     # get circles
#     circles = cv2.HoughCircles(frame, method=cv2.HOUGH_GRADIENT, dp=1,
#                                minDist=frame.shape[1], param1=50, param2=7,
#                                minRadius=1, maxRadius=100)
#     # draw circles
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
#         for (x, y, r) in circles:
#             area = \
#                 (x, y, r, frame, result)
#
#     return area

def dilate_and_erode(frame, dilation_iterations, erode_iterations, kernel_size):
    if not isinstance(frame, np.ndarray):
        return None
    kernel = np.ones((kernel_size, kernel_size))
    frame_dilated = cv2.dilate(frame, kernel, iterations=dilation_iterations)
    frame_eroded = cv2.erode(frame_dilated, kernel, iterations=erode_iterations)
    return frame_eroded


def erode_and_dilate(frame, dilation_iterations, erode_iterations, kernel_size):
    if not isinstance(frame, np.ndarray):
        return None
    kernel = np.ones((kernel_size, kernel_size))
    frame_eroded = cv2.erode(frame, kernel, iterations=erode_iterations)
    frame_dilated = cv2.dilate(frame_eroded, kernel, iterations=dilation_iterations)
    return frame_dilated


def add_text_to_screen(frame, string):
    if not isinstance(frame, np.ndarray):
        return None

    # todo: here we would add the logic if there is a goal
    cv2.putText(frame, string, org=(50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7, color=(100, 0, 255), thickness=2)


def get_ball_mask(frame):
    if not isinstance(frame, np.ndarray):
        return None
    # frame_blurred = cv2.GaussianBlur(frame, (33, 33), 1)
    # frame_blurred = cv2.medianBlur(frame_blurred, 7)
    # mask = cv2.inRange(cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV), COLOR_MIN, COLOR_MAX)
    # frame_dilated_and_eroded = dilate_and_erode(mask, 2, 2, 5)
    # # cv2.imshow("d&e",frame_dilated_and_eroded)
    # frame_canny = cv2.Canny(frame_dilated_and_eroded, threshold1, threshold2)
    # # cv2.imshow("frame_canny",frame_canny)
    # return frame_canny
    blurred = cv2.GaussianBlur(frame, (33, 33), 1)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)
    mask = erode_and_dilate(mask, 2, 2, 5)
    # frame_canny = cv2.Canny(mask, threshold1, threshold2)
    cv2.imshow("mask", mask)
    return mask


# def is_goal(center, radius, p1, p2, left_to_right, down_to_up, is_goal_horizontal):
#     x_center, y_center = center
#     x1, y1 = p1
#     x2, y2 = p2
#     if np.logical_or(np.logical_and(x_center + radius > x1, x_center + radius > x2, is_goal_horizontal,
#
#                                     left_to_right == True),
#                      np.logical_and(x_center - radius < x1, x_center - radius < x2, is_goal_horizontal,
#                                     left_to_right == False),
#                      (np.logical_and(y_center + radius > y1, y_center + radius > y2, is_goal_horizontal == False,
#                                      down_to_up == True),
#                       np.logical_and(y_center - radius < y1, y_center - radius < y2, is_goal_horizontal == False,
#                                      down_to_up == False))):
#         print("GOAL")
#     else:
#         print("NO-GOAL")
def is_goal(center, radius, r, theta, left_to_right):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * r
    y0 = b * r
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    x1, y1 = pt1
    x2, y2 = pt2
    x_center, y_center = center
    if left_to_right:
        if (x_center - radius) > x1:
            if (x_center - radius) > x2:
                goal = "GOAL"
            else:
                goal = "NO GOAL"
        else:
            goal = "NO GOAL"

    return goal
