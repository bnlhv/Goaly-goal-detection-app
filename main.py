from logicMethods import *
from drawingMethods import *





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
            ball_center, ball_radius = get_center_and_radius(ball_mask.copy())
            if ball_radius > 10:
                draw_ball(ball_center, ball_radius, result)
                #TODO : here we would pass radius and center to is_goal function
            print("----")
            print(ball_center)
            print(ball_radius)

            if is_first_frame:
                r, theta = get_goal_lines(result, cv2.Canny(frame.copy(), threshold1, threshold2), is_goal_horizontal=True,
                                      threshold_for_lines=200)
                is_first_frame = False

            draw_goal_lines(r, theta, result, is_goal_horizontal=True)
            goal_result = is_goal(r, theta, ball_radius, ball_center)
            add_text_to_screen(result, goal_result)


            cv2.imshow("Result", result)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break


main()
