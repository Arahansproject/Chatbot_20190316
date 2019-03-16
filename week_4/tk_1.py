import tkinter as tk
class Label:
    def __init__(self, text, padx, pady):
        root = tk.Tk()
        label = tk.Label(root, text=text, padx=padx, pady=pady)
        label.pack()
        root.mainloop()

def main():
    calc = Calc(int(input("첫번째 수")),int(input("두번째 수")))
    print(calc.first)
    print(calc.second)
    print("{}+{}={}".format(calc.first, calc.second, calc.sum()))
    print("{}-{}={}".format(calc.first, calc.second, calc.sub()))
    print("{}*{}={}".format(calc.first, calc.second, calc.mul()))
    print("{}/{}={}".format(calc.first, calc.second, calc.div()))

"""
OOP에서는 
선언(declaration)과 실행(execution)이 구분된다.
"""

if __name__=="__main__":
    main()