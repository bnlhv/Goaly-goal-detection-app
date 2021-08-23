import cv2
import numpy as np
import math
import imutils

COLOR_MIN = np.array([0, 15, 10], np.uint8)
COLOR_MAX = np.array([7, 255, 255], np.uint8)


def empty(a):
    pass


def my_hough_lines(canny_frame):
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


def get_lines_from_accumulator(H, r_max, r_array, t_array, threshold_for_lines):
    yy, xx = np.nonzero(H)
    r = []
    t = []
    for x, y in zip(xx, yy):
        if H[y, x] >= threshold_for_lines:
            r.append(r_array[y])
            t.append(t_array[x])
    return r, t


def get_goal_lines(frame, frame_canny, is_goal_horizontal, threshold_for_lines):
    if not isinstance(frame, np.ndarray):
        return None
    if not isinstance(frame_canny, np.ndarray):
        return None

    # get lines
    # lines = cv2.HoughLines(frame_canny, 1, np.pi / 180, threshold_for_lines, None, 0, 0)

    H, r_max, r_array, t_array = my_hough_lines(frame_canny)

    # draw lines
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


def is_goal(r, theta, ball_radius, ball_center):
    pass