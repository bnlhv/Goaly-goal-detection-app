"""
this module is responsible for all the logic functions
"""

from typing import Any, List, Union, Optional
import cv2
import numpy as np
import math
import imutils
from drawingMethods import *
from matplotlib import pyplot as plt

# these are HSV mask values for ball detection
COLOR_MIN = np.array([0, 115, 50], np.uint8)
COLOR_MAX = np.array([7, 255, 255], np.uint8)


def empty(a: Any) -> None:
    """
    this is an empty function to be able to do nothing but with more readability

    :param a: optional param
    """
    pass


def open_parameters_config_window() -> None:
    """ this function is for new window with trackbars which are responsible for app thresholds etc. """

    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", (640, 120))
    cv2.createTrackbar("Threshold1", "Parameters", 50, 255, empty)
    cv2.createTrackbar("Threshold2", "Parameters", 150, 255, empty)
    cv2.createTrackbar("Area", "Parameters", 1300, 15000, empty)


def is_line_horizontal(is_goal_horizontal: bool, theta: float) -> bool:
    """
    this function returns a value that detects if the line we found is horizontal

    :param is_goal_horizontal: bool value, what the user entered
    :param theta: the angle in pi values
    :return: boolean
    """
    # if the degree of the line is horizontal return true (4 cases)
    return np.logical_or(
        np.logical_and(is_goal_horizontal, np.logical_or(0 * np.pi <= theta <= 0.1 * np.pi,
                                                         1.9 * np.pi <= theta <= 2 * np.pi)),
        np.logical_and(is_goal_horizontal, np.logical_or(0.9 * np.pi <= theta <= 1 * np.pi,
                                                         1 * np.pi <= theta <= 1.1 * np.pi))
    )


# def is_line_vertical(is_goal_horizontal: bool, theta: float) -> bool:
#     """
#         this function returns a value that detects if the line we found is vertical
#
#         :param is_goal_horizontal: bool value, what the user entered
#         :param theta: the angle in pi values
#         :return: boolean
#         """
#     # if the degree of the line is vertical return true
#     return np.logical_and(not is_goal_horizontal, np.logical_or(0.4 * np.pi <= theta <= 0.6 * np.pi,
#                                                                 1.4 * np.pi <= theta <= 1.6 * np.pi))


def my_hough_lines(canny_frame: np.ndarray) -> Union[np.ndarray, int, float]:
    """
    this function finds the accumulator matrix and from it we can collect the lines by threshold

    :param canny_frame: ndarray that represents the Canny Edge detection image
    :return: the accumulator mat, max value of rho which is the distance
    """

    w, h = canny_frame.shape
    r_max: int = int(np.sqrt(w ** 2 + h ** 2))
    rho_list: list = np.arange(-r_max, r_max)
    theta_list: list = np.arange(np.pi, step=np.pi / 180)

    # Hough accumulator array of theta vs rho
    H: np.ndarray = np.zeros((len(rho_list), len(theta_list)), dtype=np.uint64)  # accumulator matrix
    yy, xx = np.nonzero(canny_frame)  # (row, colo) indexes to edges

    # Vote in the Hough accumulator
    t = np.zeros(len(theta_list))
    r = np.zeros(len(rho_list))

    for x, y in zip(xx, yy):
        for theta_idx, theta in enumerate(theta_list):
            if is_line_horizontal(True, theta):
                rho = x * np.cos(theta) + y * np.sin(theta)
                rho_idx = int(round(rho)) + r_max
                H[rho_idx, theta_idx] += 1
                t[theta_idx] = theta
                r[rho_idx] = rho

    return H, r, t


# todo: why not np.argmax?
def most_frequent(lst: List) -> int:
    """
    this function find the most frequent element

    :param lst:list
    :return: most frequent element
    """
    counter: int = 0
    num: int = lst[0]

    for i in lst:
        curr_frequency = lst.count(i)
        # print(curr_frequency)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


def merge_lines(rho: np.ndarray, theta: np.ndarray, left_to_right: bool) -> Union[int, float]:
    """
    get several lines that could be a goal line and merge them to one to find out if goal

    :param rho: list of rho values of the founded nominee lines
    :param theta: list of theta degrees values of the founded nominee lines
    :param left_to_right: bool to know what logic to implement
    :return: the rho, theta combination for 1 line
    """
    if np.logical_and(len(rho) == 1, len(theta) == 1):
        return rho, theta

    most_common_t: int = most_frequent(theta)

    for i, value in enumerate(theta):
        if value != most_common_t:  # somthing went worng here because there is values in t that not equal to most_coomon_t
            theta.remove(value)
            rho.remove(rho[i])

    maximum: float = max(rho)
    minimum: float = min(rho)

    if left_to_right:
        new_r = maximum
    else:
        new_r = minimum

    return new_r, most_common_t


def get_lines_from_accumulator(H, r_array, t_array, threshold_for_lines) -> Union[int, float]:
    """
    this function return the rhos and thetas which higher than treshold_for_lines

    :param H: np.ndarray ,the accumulator matrix
    :param r_array: list of rhos
    :param t_array: list of thetas
    :param threshold_for_lines: int , Sets the threshold for the lines
    :return:
    """
    if np.logical_or(r_array is None, t_array is None):
        return None
    yy, xx = np.nonzero(H)
    r = []
    t = []
    for x, y in zip(xx, yy):
        if H[y, x] >= threshold_for_lines:
            r.append(r_array[y])
            t.append(t_array[x])
    rho, theta = merge_lines(r, t, left_to_right=True)
    return rho, theta


def get_goal_lines(frame_canny, threshold_for_lines):
    """
    this function return rho, theta of the goal line

    :param frame_canny: np.ndarray of edges
    :param threshold_for_lines: int , Sets the threshold for the lines

    """
    if not isinstance(frame_canny, np.ndarray):
        return None

    H, r_array, t_array = my_hough_lines(frame_canny)

    return get_lines_from_accumulator(H, r_array, t_array, threshold_for_lines)


def get_center_and_radius(mask) -> Union[(int, int), float]:
    """
    this function recognize the center and the radius of the ball

    :param mask: np.ndarray mask that helps to find the ball
    :return: (int, int), float :center of the ball ,radius of the ball
    """
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


def erode_and_dilate(frame: np.ndarray, dilation_iterations: int, erode_iterations: int,
                     kernel_size: int) -> np.ndarray:
    """
    this function dose dilation_iterations times cv2.erode()  and erode_iterations times tomes cv2.dilate()

    :param frame: np.ndarray
    :param dilation_iterations: int of iterations for dilate
    :param erode_iterations: int of iterations for erode
    :param kernel_size: int
    :return: np.ndarray
    """
    if not isinstance(frame, np.ndarray):
        return None
    kernel = np.ones((kernel_size, kernel_size))
    frame_eroded = cv2.erode(frame, kernel, iterations=erode_iterations)
    frame_dilated = cv2.dilate(frame_eroded, kernel, iterations=dilation_iterations)
    return frame_dilated


def get_ball_mask(frame: np.ndarray) -> np.ndarray:
    """
    get the original frame and return mask the helps to find the ball

    :param frame: np.ndarray , original frmae
    :return: np.ndarray , mask for the ball
    """
    if not isinstance(frame, np.ndarray):
        return None
    blurred = cv2.GaussianBlur(frame, (33, 33), 1)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)
    mask = erode_and_dilate(mask, 2, 2, 5)
    cv2.imshow("mask2", mask)
    return mask


def is_goal(center: (int, int), radius: float, r: int, theta: float, left_to_right: bool) -> str:
    """
    this function determines in each frame if it is a goal or not and return string

    :param center:(x,y) point of the ball center
    :param radius: radius of the ball
    :param r: rhos of the goal line
    :param theta: theta of the goal line
    :param left_to_right: if True it is mean the goal come from left to right, if False its means the opposite
    :return: goal or no goal
    """
    if isinstance(r, List):
        r = r[0]
        theta = theta[0]

    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * r
    y0 = b * r
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    x1, y1 = pt1
    x2, y2 = pt2
    x_center, y_center = center
    d = float((x_center - radius - x1) * (y2 - y1) - (y_center - y1) * (x2 - x1))

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

    return goal
