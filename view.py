""" This Module is responsible for the view - User Interface """

from tkinter import Tk, filedialog, Button, Frame, Radiobutton, Label, IntVar, BOTTOM, LEFT, RIGHT
from app import app

# Global variables for this python file
root = Tk()
left_to_right = IntVar()


def open_file() -> None:
    """
    This function allow to user open file from pc
    """
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select a video", filetypes=[("mp4", "*.mp4")])
    filedialog.path_label.config(text=root.filename)
    button2 = Button(root, text="start", font=11, fg="blue", command=start_goal_detection, height=2, width=5)
    button2.place(relx=0.5, rely=0.5, anchor='center')
    button2.pack()


def start_goal_detection() -> None:
    """
    This function run the goal detection algo
    """
    if left_to_right.get() == 1:
        ltr = True
    elif left_to_right.get() == 2:
        ltr = False
    else:
        ltr = None
    app(root.filename, ltr)


def render_view() -> None:
    """
    This is the main UI function, renders the window
    """
    root.title("GOALY")
    root.iconbitmap("goaly.ico")
    root.geometry('500x300')
    root.eval('tk::PlaceWindow . center')

    top_frame = Frame(root)
    top_frame.pack()
    bottom_frame = Frame(root)
    bottom_frame.pack(side=BOTTOM)

    path_label = Label(bottom_frame, bg="white", width=50, text='  ', font=("Courier", 8))
    path_label.pack(side=LEFT)

    button1 = Button(bottom_frame, text="Open A Video", fg="red", command=open_file, font=4)
    button1.pack(side=RIGHT)

    left_to_right.set(1)

    label = Label(root, width=40, text='Select the direction of the goal :', font=4)
    label.pack()

    Radiobutton(root, text='Left to right', font=3, variable=left_to_right,
                command=lambda: left_to_right.set(1), value=1).pack()

    Radiobutton(root, text='Right to left', font=3, variable=left_to_right,
                command=lambda: left_to_right.set(2), value=2).pack()

    root.mainloop()
