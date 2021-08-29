"""
this module is responsible for all the drawing functions
"""

from logicMethods import *


def draw_line(frame: np.ndarray, theta: List, rho: float) -> None:
    """
    this function draw a line on a canvas "frame" with converting rho, theta coefficient values to [y = ax + b]

    :param frame: nd array -> the current picture at the video
    :param theta: angle which is an axis is hough space
    :param rho: distance from (0, 0) to (x, y) which is an axis is hough space
    """
    if isinstance(rho, List):
        rho = rho[0]
        theta = theta[0]

    if rho is not None:
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
        pt2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
        cv2.line(frame, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)


def draw_ball(center: (int, int), radius: float, result: np.ndarray) -> None:
    """
    this function draw a circle on a canvas "result"

    :param center: (int,int) -> (x,y) point of the ball center
    :param radius: float -> radius of the ball
    :param result: nd array -> it is the frame on which the circle is drawn
    """
    cv2.circle(result, center, int(radius), (0, 255, 0), 2)


def add_text_to_screen(frame: np.ndarray, string: str) -> None:
    """
    this function write the string on the frame

    :param frame:nd array -> the current picture at the video
    :param string: this string is written on the frame
    """
    if not isinstance(frame, np.ndarray):
        return None

    cv2.putText(frame, string, org=(50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7, color=(100, 0, 255), thickness=2)
