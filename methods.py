import cv2
import numpy as np
import math


WHITE_MIN = np.array([0, 0, 50], np.uint8)
WHITE_MAX = np.array([40, 40, 185], np.uint8)


def empty(a):
    pass


def open_parameters_config_window():
    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", (640, 120))
    cv2.createTrackbar("Threshold1", "Parameters", 50, 255, empty)
    cv2.createTrackbar("Threshold2", "Parameters", 150, 255, empty)
    cv2.createTrackbar("Area", "Parameters", 1300, 15000, empty)


def draw_line(frame, theta, rho):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    cv2.line(frame, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)


def is_line_vertical(is_goal_horizontal, theta):
    # if the degree of the line is vertical return true
    return np.logical_and(not is_goal_horizontal, np.logical_or((1/3) * np.pi <= theta <= (2 / 3) * np.pi,
                                                                (4 / 3) * np.pi <= theta <= (5 / 3) * np.pi))


def is_line_horizontal(is_goal_horizontal, theta):
    # if the degree of the line is horizontal return true (4 cases)
    return np.logical_or(
                np.logical_and(is_goal_horizontal, np.logical_or(0 * np.pi <= theta <= (1 / 6) * np.pi,
                                                                (11 / 6) * np.pi <= theta <= 2 * np.pi)),
                np.logical_and(is_goal_horizontal, np.logical_or((5 / 6) * np.pi <= theta <= 1 * np.pi,
                                                                 1 * np.pi <= theta <= (7 / 6) * np.pi))
                        )


def draw_goal_lines(lines, frame, is_goal_horizontal):
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            if is_line_vertical(is_goal_horizontal, theta):
                draw_line(frame, theta, rho)

            elif is_line_horizontal(is_goal_horizontal, theta):
                draw_line(frame, theta, rho)

            else:
                empty(0)


def get_goal_lines(frame, frame_canny, is_goal_horizontal, threshold_for_lines):
    # todo:Here we would create our own hough lines
    # get lines
    lines = cv2.HoughLines(frame_canny, 1, np.pi / 180, threshold_for_lines, None, 0, 0)

    # draw lines
    draw_goal_lines(lines, frame, is_goal_horizontal)


def draw_ball_contours(contour, frame, frame_contours):
    if cv2.isContourConvex(contour):
        frame_contours = cv2.convexHull(contour)
    else:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if np.pi * (radius ** 2) < 0.05 * frame.shape[0] * frame.shape[1]:
            center = (int(x), int(y))
            radius = int(radius)
            frame_contours = cv2.circle(frame_contours, center, radius, (0, 255, 0), 2)


def get_ball_contours(frame, frame_contours):
    # get contours
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours
    for contour in contours:
        area = cv2.contourArea(contour)
        area_from_user = cv2.getTrackbarPos("Area", "Parameters")
        if area > area_from_user:
            draw_ball_contours(contour, frame, frame_contours)

    return area


def draw_ball_hough_circle(x, y, r, frame, result):
    area_from_user = cv2.getTrackbarPos("Area", "Parameters")
    area = np.pi * (r ** 2)
    if area_from_user < area < (0.06 * frame.shape[0] * frame.shape[1]):
        cv2.circle(result, (x, y), r, (255, 0, 0), 3)

    return area


def get_ball_hough_circles(frame, result):
    # get circles
    circles = cv2.HoughCircles(frame, method=cv2.HOUGH_GRADIENT , dp=1,
                               minDist=frame.shape[1], param1=50, param2=7,
                               minRadius=1, maxRadius=100)
    # draw circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            area = draw_ball_hough_circle(x, y, r, frame, result)

    return area


def dilate_and_erode(frame, dilation_iterations, erode_iterations, kernel_size):
    kernel = np.ones((kernel_size, kernel_size))
    frame_dilated = cv2.dilate(frame, kernel, iterations=dilation_iterations)
    frame_eroded = cv2.erode(frame_dilated, kernel, iterations=erode_iterations)
    return frame_eroded


def erode_and_dilate(frame, dilation_iterations, erode_iterations, kernel_size):
    kernel = np.ones((kernel_size, kernel_size))
    frame_eroded = cv2.erode(frame, kernel, iterations=erode_iterations)
    frame_dilated = cv2.dilate(frame_eroded, kernel, iterations=dilation_iterations)
    return frame_dilated


def add_text_to_screen(frame, area):
    # todo: here we would add the logic if there is a goal
    cv2.putText(frame, "Here we would write if there is a goal", org=(50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7, color=(100, 0, 255), thickness=2)
    cv2.putText(frame, "Area: " + str(int(area)), org=(50, 100), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7, color=(100, 0, 255), thickness=2)


def get_canny_frame(frame, threshold1, threshold2):
    frame_blurred = cv2.GaussianBlur(frame, (33, 33), 1)
    frame_blurred = cv2.medianBlur(frame_blurred, 7)
    mask = cv2.inRange(cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV), WHITE_MIN, WHITE_MAX)
    frame_dilated_and_eroded = dilate_and_erode(mask, 2, 1, 5)
    frame_canny = cv2.Canny(frame_dilated_and_eroded, threshold1, threshold2)

    return frame_canny
