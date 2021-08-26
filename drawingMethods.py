# from logicMethods import *
#
#
# def open_parameters_config_window():
#     cv2.namedWindow("Parameters")
#     cv2.resizeWindow("Parameters", (640, 120))
#     cv2.createTrackbar("Threshold1", "Parameters", 50, 255, empty)
#     cv2.createTrackbar("Threshold2", "Parameters", 150, 255, empty)
#     cv2.createTrackbar("Area", "Parameters", 1300, 15000, empty)
#
#
# def draw_line(frame, theta, rho):
#     if not isinstance(frame, np.ndarray):
#         return None
#     a = math.cos(theta)
#     b = math.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     pt1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
#     pt2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
#     print(f"r = {rho}, theta = {theta}, p1 = {pt1}, p2 = {pt2}")
#     cv2.line(frame, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
#
#
# def is_line_vertical(is_goal_horizontal, theta):
#     # if the degree of the line is vertical return true
#     return np.logical_and(not is_goal_horizontal, np.logical_or(0.4 * np.pi <= theta <= 0.6 * np.pi,
#                                                                 1.4 * np.pi <= theta <= 1.6 * np.pi))
#
#
# def is_line_horizontal(is_goal_horizontal, theta):
#     # if the degree of the line is horizontal return true (4 cases)
#     return np.logical_or(
#         np.logical_and(is_goal_horizontal, np.logical_or(0 * np.pi <= theta <= 0.1 * np.pi,
#                                                          1.9 * np.pi <= theta <= 2 * np.pi)),
#         np.logical_and(is_goal_horizontal, np.logical_or(0.9 * np.pi <= theta <= 1 * np.pi,
#                                                          1 * np.pi <= theta <= 1.1 * np.pi))
#     )
#
#
# def draw_goal_lines(r, theta, frame, is_goal_horizontal):
#     if not isinstance(frame, np.ndarray):
#         return None
#
#     if r is not None:
#         draw_line(frame, theta, r)
#
#
# def draw_ball(center, radius, result):
#     cv2.circle(result, center, int(radius), (0, 255, 0), 2)
#
#
# def add_text_to_screen(frame, area):
#     if not isinstance(frame, np.ndarray):
#         return None
#     if not isinstance(area, np.ndarray):
#         return None
#     # todo: here we would add the logic if there is a goal
#     cv2.putText(frame, "Here we would write if there is a goal", org=(50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX,
#                 fontScale=0.7, color=(100, 0, 255), thickness=2)
#     cv2.putText(frame, "Area: " + str(int(area)), org=(50, 100), fontFace=cv2.FONT_HERSHEY_COMPLEX,
#                 fontScale=0.7, color=(100, 0, 255), thickness=2)