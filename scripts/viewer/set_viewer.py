import Tkinter as tk
import argparse

import numpy as np

from storage.candidate_iterator import CandidateIter
from arrayviewer import Array3DViewer


class SetViewer(tk.Frame):
    def __init__(self, master = None, root = "", subsets = (-1,)):
        tk.Frame.__init__(self, master)
        self.grid(padx = 2, pady = 2)
        self.root = root
        self.frame = self.make_frame()
        self.button = tk.Button(self.frame, text = "Next Sample", command = self._next)
        self.button.pack(side = tk.LEFT)
        self.label = tk.Label(self.frame, text = "-")
        self.label.pack(side = tk.RIGHT)
        self.viewer = Array3DViewer(self, row = 1)
        self.iterator = CandidateIter(self.root, subsets, batch_size = 1)

        self.positive_only_var = tk.IntVar()
        self.positive_only_var.set(True)
        self.positive_only_menu = tk.Checkbutton(self.frame, variable = self.positive_only_var)
        self.positive_only_menu.pack(side = tk.RIGHT, padx = 5)
        self.positive_only_label = tk.Label(self.frame, text = "Positive only?")
        self.positive_only_label.pack(side = tk.RIGHT, padx = 5)

    def make_frame(self, row = 0, column = 0, rowspan = 1, columnspan = 1):
        frame = tk.Frame(self, borderwidth = 3, relief = "ridge")
        frame.grid(row = row, column = column, rowspan = rowspan, columnspan = columnspan, padx = 2, pady = 2, sticky = tk.E + tk.W + tk.S + tk.N)
        return frame

    def _next(self):
        while True:
            batch = self.iterator.next()
            array = np.squeeze(batch.data[0].asnumpy()[0])
            label = "ok"
            if batch.label[0].asnumpy()[0] > 0.5:
                label = "malignant"
                if self.positive_only_var.get():
                    break
            if not self.positive_only_var.get():
                break
        self.label.configure(text = label)
        self.viewer.set_array(array, np.asarray((0, 0, 0)), np.asarray((1, 1, 1)))


def main(args):
    viewer = SetViewer(master = None, root = args.root, subsets = args.subsets)
    viewer.master.title('Viewer')
    viewer.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "show the contents of a prepared 3D array")
    parser.add_argument("root", type=str, help="folder containing prepared dataset folders")
    parser.add_argument("--subsets", type=int, nargs="*", help="the subsets which should be processed", default = range(0, 10))
    main(parser.parse_args())
