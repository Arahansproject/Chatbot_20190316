from tkinter import *


class CalUI:

    def __init__(self):
        pass

    @staticmethod
    def main():
        root = Tk()
        root.title("계산기")
        w = 300
        h = 200
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        root.geometry("%dx%d+%d+%d" % (w, h, ws, hs))
        root.resizable(FALSE, FALSE)
        root.geometry("380x200")
        root.mainloop()




