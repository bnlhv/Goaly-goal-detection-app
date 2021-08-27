"""this module is main entry point of this app"""

from logicMethods import *


def goal_detection_app():

    cap = cv2.VideoCapture("C:\\Users\\benla\\Downloads\\demo16.mp4")
    is_first_frame = True
    while True:
        success, frame = cap.read()

        if success:
            frame = cv2.resize(frame, (360, 640))
            frame = frame[5: -5, 5: -5]
            result = frame.copy()

            if is_first_frame:
                r, theta = get_goal_lines(result, cv2.Canny(frame.copy(), 50, 150),
                                          threshold_for_lines=200)
                is_first_frame = False

            draw_line(result, theta, r)
            ball_mask = get_ball_mask(frame.copy())
            center, radius = get_center_and_radius(ball_mask.copy())

            if radius > 15:
                draw_ball(center, radius, result)
                goal = is_goal(center, radius, r, theta, left_to_right=True)
                add_text_to_screen(result, goal)

            cv2.imshow("Result", result)

            k = cv2.waitKey(50) & 0xff
            if k == 27:
                break
        else:
            break


if __name__ == '__main__':
    goal_detection_app()
