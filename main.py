from methods import *


def main():
    open_parameters_config_window()
    cap = cv2.VideoCapture("demo4.mp4")
    is_first_frame = True
    while True:
        success, frame = cap.read()

        if success:
            frame = cv2.resize(frame, (396, 594))
            result = frame.copy()

            threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
            threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

            if is_first_frame:
                r, theta = get_goal_lines(result, cv2.Canny(frame.copy(), threshold1, threshold2),
                                          is_goal_horizontal=True,
                                          threshold_for_lines=200)
                is_first_frame = False
            print("theta = {}".format(theta), "r = {}".format(r), "main")
            draw_goal_lines(r, theta, result, is_goal_horizontal=True)
            ball_mask = get_ball_mask(frame.copy())
            center, radius = get_center_and_radius(ball_mask.copy())

            if radius > 10:
                draw_ball(center, radius, result)
                goal = is_goal(center, radius, r, theta, left_to_right=True)
                add_text_to_screen(result, goal)

            cv2.imshow("Result", result)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break


main()
