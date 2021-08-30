"""
this module is responsible for all the logic functions
"""

import math
from typing import Optional, Tuple

import cv2
import imutils
import numpy as np

# these are HSV mask values for ball detection
COLOR_MIN = np.array([0, 120, 50], np.uint8)
COLOR_MAX = np.array([7, 255, 255], np.uint8)


def empty(a) -> None:
    """
    this is an empty function to be able to do nothing but with more readability

    :param a: optional param
    """
    pass


def is_line_horizontal(is_goal_horizontal: bool, theta: float) -> bool:
    """
    this function returns a value that detects if the line we found is horizontal

    :param is_goal_horizontal: bool value, what the user entered
    :param theta: the angle in pi values
    :return: boolean
    """
    return np.logical_or(
        np.logical_and(is_goal_horizontal, np.logical_or(0 * np.pi <= theta <= 0.1 * np.pi,
                                                         1.9 * np.pi <= theta <= 2 * np.pi)),
        np.logical_and(is_goal_horizontal, np.logical_or(0.9 * np.pi <= theta <= 1 * np.pi,
                                                         1 * np.pi <= theta <= 1.1 * np.pi))
    )


def my_hough_lines(canny_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    this function finds the accumulator matrix and from it we can collect the lines by threshold

    :param canny_frame: ndarray that represents the Canny Edge detection image
    :return: the accumulator mat, max value of rho which is the distance
    """

    w, h = canny_frame.shape
    r_max: int = int(np.sqrt(w ** 2 + h ** 2))
    rho_list: np.ndarray = np.arange(-r_max, r_max)
    theta_list: np.ndarray = np.arange(np.pi, step=np.pi / 180)

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
def most_frequent(arr: list) -> int:
    """
    this function find the most frequent element

    :param arr:list
    :return: most frequent element
    """
    counter: int = 0
    try:
        num: int = arr[0]
    except Exception as ex:
        num = arr

    for i in arr:
        curr_frequency = arr.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


def merge_lines(rho: list, theta: list, left_to_right: bool) -> Tuple[float, float]:
    """
    get several lines that could be a goal line and merge them to one to find out if goal

    :param rho: list of rho values of the founded nominee lines
    :param theta: list of theta degrees values of the founded nominee lines
    :param left_to_right: bool to know what logic to implement
    :return: the rho, theta combination for 1 line
    """
    if np.logical_and(len(rho) == 1, len(theta) == 1):
        return float(rho[0]), float(theta[0])

    most_common_t: float = most_frequent(theta)

    for i, value in enumerate(theta):
        if value != most_common_t:
            theta.remove(value)
            rho.remove(rho[i])

    maximum: float = max(rho)
    minimum: float = min(rho)

    new_r = maximum if left_to_right else minimum

    return float(new_r), float(most_common_t)


def get_line_from_accumulator(H: np.ndarray, r_array: np.ndarray, t_array: np.ndarray, threshold_for_lines: int,
                              left_to_right: bool) -> Optional[Tuple[float, float]]:
    """
    this function return the rhos and thetas which higher than treshold_for_lines

    :param left_to_right: goal direction
    :param H: np.ndarray ,the accumulator matrix
    :param r_array: ndarray of rhos
    :param t_array: ndarray of thetas
    :param threshold_for_lines: int , Sets the threshold for the lines
    :return:
    """
    if np.logical_or(r_array is None, t_array is None):
        return None

    yy, xx = np.nonzero(H)
    r, t = [], []

    for x, y in zip(xx, yy):
        if H[y, x] >= threshold_for_lines:
            r.append(r_array[y])
            t.append(t_array[x])
    rho, theta = merge_lines(r, t, left_to_right)

    return rho, theta


def get_goal_lines(frame_canny: np.ndarray, threshold_for_lines: int, left_to_right: bool) -> Optional[Tuple
    [float, float]]:
    """
    this function return rho, theta of the goal line
    :param left_to_right:
    :param frame_canny: np.ndarray of edges
    :param threshold_for_lines: int , Sets the threshold for the lines
    """
    if not isinstance(frame_canny, np.ndarray):
        return None

    H, r_array, t_array = my_hough_lines(frame_canny)

    return get_line_from_accumulator(H, r_array, t_array, threshold_for_lines, left_to_right)


def get_center_and_radius(mask) -> [(int, int), float]:
    """
    this function recognize the center and the radius of the ball

    :param mask: np.ndarray mask that helps to find the ball
    :return: (int, int), float :center of the ball ,radius of the ball
    """
    if not isinstance(mask, np.ndarray):
        return None

    center = None
    radius = 0
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    return center, radius


def erode_and_dilate(frame: np.ndarray, dilation_iterations: int, erode_iterations: int,
                     kernel_size: int) -> Optional[np.ndarray]:
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


def get_ball_mask(frame: np.ndarray) -> Optional[np.ndarray]:
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
    a, b = math.cos(theta), math.sin(theta)
    x0, y0 = a * r, b * r
    pt1, pt2 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a)), (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
    x1, y1 = pt1
    x2, y2 = pt2
    x_center, y_center = center
    d = float((x_center - radius - x1) * (y2 - y1) - (y_center - y1) * (x2 - x1))

    goal_msg = ("GOAL" if d < 0 else "NO GOAL") if left_to_right else ("GOAL" if d > 0 else "NO GOAL")

    return goal_msg
