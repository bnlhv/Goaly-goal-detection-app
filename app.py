"""
this module is responsible for goal detection
"""

from logic_functions import get_goal_lines, get_ball_mask, get_center_and_radius, is_goal
from drawing_functions import draw_line, draw_ball, add_text_to_screen
import numpy as np
import cv2


def app(file_path, left_to_right):
    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width = int(width / 3)
    height = int(height / 3)

    is_first_frame = True

    while True:
        success, frame = cap.read()

        if success:
            frame = cv2.resize(frame, (width, height))
            frame = frame[5: -5, 5: -5]
            result = frame.copy()

            if is_first_frame:
                r, theta = get_goal_lines(cv2.Canny(frame.copy(), 50, 150),
                                          threshold_for_lines=150, left_to_right=left_to_right)
                is_first_frame = False
            if np.logical_and(theta is not None, r is not None):
                draw_line(result, theta, r)
                ball_mask = get_ball_mask(frame.copy())
                center, radius = get_center_and_radius(ball_mask.copy())

            if radius > 15:
                draw_ball(center, radius, result)
                goal = is_goal(center, radius, r, theta, left_to_right=left_to_right)
                add_text_to_screen(result, goal)

            cv2.imshow("Result", result)

            k = cv2.waitKey(50) & 0xff
            if k == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
