"""This module is main run the GUI of this app"""

import tkinter
from tkinter import *
from tkinter import filedialog
from goalDetection import goal_detection_app


def open_file():
    """
    This function allow to user open file from pc
    """
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select a video", filetypes=[("mp4", "*.mp4")])
    path_label.config(text=root.filename)
    button2 = Button(root, text="start", font=11, fg="blue", command=start_goal_detection, height=2, width=5)
    button2.place(relx=0.5, rely=0.5, anchor='center')
    button2.pack()


def start_goal_detection():
    """
    This function run the goal detection algo
    """
    if left_to_right.get() == 1:
        ltr = True
    elif left_to_right.get() == 2:
        ltr = False
    else:
        ltr = None
    goal_detection_app(root.filename, ltr)


if __name__ == '__main__':
    root = Tk()
    root.title("GOALY")
    root.iconbitmap("goaly.ico")
    root.geometry('500x300')
    root.eval('tk::PlaceWindow . center')

    top_frame = Frame(root)
    top_frame.pack()
    bottom_frame = Frame(root)
    bottom_frame.pack(side=BOTTOM)

    path_label = tkinter.Label(bottom_frame, bg="white", width=50, text='  ', font=("Courier", 8))
    path_label.pack(side=LEFT)

    button1 = Button(bottom_frame, text="Open A Video", fg="red", command=open_file, font=4)
    button1.pack(side=RIGHT)

    left_to_right = tkinter.IntVar()
    left_to_right.set(1)

    label = tkinter.Label(root, width=40, text='Select the direction of the goal :', font=4)
    label.pack()

    Radiobutton(root, text='Left to right', font=3, variable=left_to_right,
                command=lambda: left_to_right.set(1), value=1).pack()

    Radiobutton(root, text='Right to left', font=3, variable=left_to_right,
                command=lambda: left_to_right.set(2), value=2).pack()

    root.mainloop()
