import cv2
import numpy as np
import math
import imutils
from matplotlib import pyplot as plt

COLOR_MIN = np.array([0, 115, 50], np.uint8)
COLOR_MAX = np.array([7, 255, 255], np.uint8)


def empty(a):
    pass


# def open_parameters_config_window():
#     cv2.namedWindow("Parameters")
#     cv2.resizeWindow("Parameters", (640, 120))
#     cv2.createTrackbar("Threshold1", "Parameters", 50, 255, empty)
#     cv2.createTrackbar("Threshold2", "Parameters", 150, 255, empty)
#     cv2.createTrackbar("Area", "Parameters", 1300, 15000, empty)

def draw_line(frame, theta, rho):
    if not isinstance(frame, np.ndarray):
        return None
    a = math.cos(theta[0])
    b = math.sin(theta[0])
    x0 = a * rho[0]
    y0 = b * rho[0]
    pt1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
    pt2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
    # print(f"r = {rho}, theta = {theta}, p1 = {pt1}, p2 = {pt2}")
    cv2.line(frame, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)


def is_line_horizontal(is_goal_horizontal, theta):
    # if the degree of the line is horizontal return true (4 cases)
    return np.logical_or(
        np.logical_and(is_goal_horizontal, np.logical_or(0 * np.pi <= theta <= 0.1 * np.pi,
                                                         1.9 * np.pi <= theta <= 2 * np.pi)),
        np.logical_and(is_goal_horizontal, np.logical_or(0.9 * np.pi <= theta <= 1 * np.pi,
                                                         1 * np.pi <= theta <= 1.1 * np.pi))
    )


def draw_goal_lines(r, theta, frame, is_goal_horizontal):
    if not isinstance(frame, np.ndarray):
        return None

    if r is not None:
        draw_line(frame, theta, r)


def my_hough_lines(canny_frame):
    # cv2.imshow("canny_frame", canny_frame)
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
            if is_line_horizontal(True, theta):
                rho = x * np.cos(theta) + y * np.sin(theta)
                rho_idx = int(round(rho)) + r_max
                hough_space[rho_idx, t_idx] += 1
                t[t_idx] = theta
                r[rho_idx] = rho
    return hough_space, r_max, r, t


def is_line_vertical(is_goal_horizontal, theta):
    # if the degree of the line is vertical return true
    return np.logical_and(not is_goal_horizontal, np.logical_or(0.4 * np.pi <= theta <= 0.6 * np.pi,
                                                                1.4 * np.pi <= theta <= 1.6 * np.pi))


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        # print(curr_frequency)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


def merge_lines(r, t, left_to_right):
    if np.logical_and(len(r) == 1, len(t) == 1):
        return r, t

    most_common_t = most_frequent(t)

    for i, value in enumerate(t):
        if value != most_common_t:  # somthing went worng here because there is values in t that not equal to most_coomon_t
            # print("value ={} =!".format(value), "most_common={}".format(most_common_t))
            t.remove(value)
            r.remove(r[i])
        # else:
        #     # print("value ={} =".format(value), "most_common={}".format(most_common_t))

    maximum = max(r)
    minimum = min(r)

    # print("theta = {}".format(t), "r = {}".format(r))
    if left_to_right:
        new_r = maximum
    else:
        new_r = minimum

    # print("new_r = {}".format(new_r), "new_t = {}".format(most_common_t))
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


def get_goal_lines(frame, frame_canny, threshold_for_lines):
    if not isinstance(frame, np.ndarray):
        return None
    if not isinstance(frame_canny, np.ndarray):
        return None

    H, r_max, r_array, t_array = my_hough_lines(frame_canny)

    return get_lines_from_accumulator(H, r_max, r_array, t_array, threshold_for_lines)


def get_center_and_radius(mask):
    if not isinstance(mask, np.ndarray):
        return None
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
    blurred = cv2.GaussianBlur(frame, (33, 33), 1)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)
    mask = erode_and_dilate(mask, 2, 2, 5)
    cv2.imshow("mask2", mask)
    return mask


def is_goal(center, radius, r, theta, left_to_right):
    goal = ""
    a = math.cos(theta[0])
    b = math.sin(theta[0])
    x0 = a * r[0]
    y0 = b * r[0]
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    x1, y1 = pt1
    x2, y2 = pt2
    x_center, y_center = center
    d = float((x_center - radius - x1) * (y2 - y1) - (y_center - y1) * (x2 - x1))
    print(d)

    if left_to_right:
        if d < 0:
            goal = "GOAL"
        else:
            goal = "NO GOAL"
    else:
        if d > 0:
            goal = "GOAL"
        else:
            goal = "NO GOAL"
    print(goal)
    return goal
