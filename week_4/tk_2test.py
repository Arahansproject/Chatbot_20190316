from tkinter import *


class CalUI:

    def __init__(self):
        pass

    @staticmethod
    def main():
        root = Tk()
        root.title("계 산 기")
        w = 300
        h = 200
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        root.geometry("%dx%d+%d+%d" % (w, h, x, y))
        root.resizable(FALSE, FALSE)
        root.geometry("380x200")

        root.mainloop()


if __name__ == '__main__':
    CalUI.main()