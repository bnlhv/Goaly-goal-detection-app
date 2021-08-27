"""this module is main entry point of this app"""

from logicMethods import *


def goal_detection_app():
    cap = cv2.VideoCapture("demo18.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # FPS = cap.get(cv2.CAP_PROP_FPS)

    width = int(width / 3)
    height = int(height / 3)


    is_first_frame = True

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # file_name = 'output1.mp4'
    # resolution = (width, height)
    # video_output = cv2.VideoWriter(file_name, fourcc, FPS, resolution)

    while True:
        success, frame = cap.read()

        if success:
            frame = cv2.resize(frame, (width, height))
            frame = frame[5: -5, 5: -5]
            result = frame.copy()

            if is_first_frame:
                r, theta = get_goal_lines(cv2.Canny(frame.copy(), 50, 150),
                                          threshold_for_lines=200)
                is_first_frame = False

            draw_line(result, theta, r)
            ball_mask = get_ball_mask(frame.copy())
            center, radius = get_center_and_radius(ball_mask.copy())

            if radius > 15:
                draw_ball(center, radius, result)
                goal = is_goal(center, radius, r, theta, left_to_right=True)
                add_text_to_screen(result, goal)

            # video_output.write(result)
            cv2.imshow("Result", result)

            k = cv2.waitKey(50) & 0xff
            if k == 27:
                break
        else:
            break
    # video_output.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    goal_detection_app()
