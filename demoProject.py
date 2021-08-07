import cv2
import numpy as np
import math

lower = np.array([160, 100, 100])
upper = np.array([180, 255, 255])

# todo: try with camera
cap = cv2.VideoCapture('20210807_132527.mp4')

while (1):
    ret, frame = cap.read() #read this frame and move to next

    if ret == True:
        frame = cv2.resize(frame, (640, 360))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(frame, 100, 200, None, 3)

        # lines
        lines = cv2.HoughLines(canny, 1, np.pi / 180, 170, None, 0, 0)

        # blur = cv2.GaussianBlur(frame_gray ,(0,0),5)
        # circles
        _, bw = cv2.threshold(frame_gray, 150, 200, cv2.THRESH_BINARY_INV)

        cv2.imshow('bw', bw)  # display frame in a window

        circles = cv2.HoughCircles(frame_gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=frame.shape[0], param1=100, param2=2,
                                   minRadius=0, maxRadius=50)
        circles = circles.astype(np.int)

        # draw circles
        for (x, y, r) in circles[0]:
            cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
        # draw lines
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                # todo: maybe try to analyze the pixel color -> if 200-250 then we know its white and probably the
                #  goaline
                if 0*np.pi <= theta <= 0.20*np.pi or 0.8*np.pi <= theta <= np.pi:
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)


        cv2.imshow('final', frame)  # display frame in a window

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    else:
        break

print(bw.__class__)
print(circles.__class__)

cap.release()  # release input video
cv2.destroyAllWindows()  # delete output window
cv2.waitKey(1);