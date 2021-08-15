from methods import *


def main():
    open_parameters_config_window()
    cap = cv2.VideoCapture("20210807_133945.mp4")

    while True:
        success, frame = cap.read()

        if success:
            frame = cv2.resize(frame, (640, 360))
            result = frame.copy()

            threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
            threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

            frame_canny = get_canny_frame(frame.copy(), threshold1, threshold2)

            ball_area = get_ball_contours(frame_canny.copy(), result)
            get_goal_lines(result, frame_canny.copy(), is_goal_horizontal=True, threshold_for_lines=100)

            add_text_to_screen(result, ball_area)

            cv2.imshow("Result", result)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break


main()
