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

            ball_mask = get_ball_mask(frame.copy())
            center, radius = get_center_and_radius(ball_mask.copy())
            if radius > 10:
                draw_ball(center, radius, result)
                #TODO : here we would pass radius and center to is_goal function
            print("----")
            print(center)
            print(radius)

            if is_first_frame:
                r, t = get_goal_lines(result, cv2.Canny(frame.copy(), threshold1, threshold2), is_goal_horizontal=True,
                                      threshold_for_lines=200)
                is_first_frame = False
            draw_goal_lines(r, t, result, is_goal_horizontal=True)
            # add_text_to_screen(result, ball_area)
            cv2.imshow("Result", result)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break


main()
