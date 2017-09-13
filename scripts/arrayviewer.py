import helper, argparse, os
from PIL import Image, ImageTk
import Tkinter as tk
import numpy as np


class Array3DViewer(tk.Frame):
    def __init__(self, master = None, canvas_size = 512, stretch = True, row = 0, column = 0, rowspan = 1, columnspan = 1):
        tk.Frame.__init__(self, master)

        self.canvas_size = canvas_size
        self.stretch = stretch

        self.grid(padx = 2, pady = 2, row = row, column = column, rowspan = rowspan, columnspan = columnspan)
        self.create_widgets()
        self.bind_all("<Right>", self.on_right_pressed)
        self.bind_all("<Left>", self.on_left_pressed)

        self.connections = []

    def connect_on_coordinate_changed(self, callback):
        self.connections.append(callback)

    def create_widgets(self):
        self.create_coordinates()
        self.create_mouse_field()
        self.create_canvas()

    def make_frame(self, row = 0, column = 0, rowspan = 1, columnspan = 1):
        frame = tk.Frame(self, borderwidth = 3, relief = "ridge")
        frame.grid(row = row, column = column, rowspan = rowspan, columnspan = columnspan, padx = 2, pady = 2, sticky = tk.E + tk.W + tk.S + tk.N)
        return frame

    def create_coordinates(self):
        self.coordinate_frame = self.make_frame(row = 2, column = 1)

        self.coordinate_label = tk.Label(self.coordinate_frame, text = "")
        self.coordinate_label.pack(side = tk.LEFT, padx = 5)
        self.coordinate_var = tk.IntVar()
        self.coordinate_var.set(0)
        self.coordinate_menu = tk.Scale(self.coordinate_frame, variable = self.coordinate_var, showvalue = False, orient=tk.HORIZONTAL, command=self.on_coordinate_changed, length = 200)
        self.coordinate_menu.pack(side = tk.RIGHT, padx = 5)

    def create_mouse_field(self):
        self.mouse_frame = self.make_frame(row = 2, column = 0)
        self.mouse_label = tk.Label(self.mouse_frame, text = "X: -, Y: -")
        self.mouse_label.pack(side = tk.LEFT, padx = 5)

    def create_canvas(self):
        self.canvas_frame = self.make_frame(row = 3, column = 0, columnspan = 2)
        self.canvas = tk.Canvas(self.canvas_frame, width = self.canvas_size, height = self.canvas_size)
        self.canvas.pack(side = tk.BOTTOM)
        self.canvas.bind('<Motion>', self.on_mouse_movement)
        self.canvas.bind('<Leave>', self.on_mouse_leave)

        self.image_on_canvas = None

    def on_right_pressed(self, event):
        self.coordinate_var.set(self.coordinate_var.get() + 1)
        self.on_coordinate_changed()

    def on_left_pressed(self, event):
        self.coordinate_var.set(self.coordinate_var.get() - 1)
        self.on_coordinate_changed()

    def on_mouse_movement(self, event):
        if not hasattr(self, 'array'):
            return

        # axis order is z, y, x, and data x and y axis are swapped
        _, y, x = helper.voxel_to_world((0, event.y * self.stretch_factor[1], event.x * self.stretch_factor[0]), self.origin, self.spacing)
        self.mouse_label.configure(text = "X: %.2f, Y: %.2f" % (x, y))

    def on_mouse_leave(self, event):
        self.mouse_label.configure(text = "X: -, Y: -")

    def on_coordinate_changed(self, _ = 0):
        if not hasattr(self, 'array'):
            return

        self.update_image(self.coordinate_var.get())
        self.coordinate_label.configure(text = "Z: %.2f" % self.get_currrent_z())

        for callback in self.connections:
            callback(self.coordinate_var.get())

    def set_array(self, array, origin, spacing):
        self.array = array
        self.origin = origin
        self.spacing = spacing

        self.coordinate_var.set(0)
        self.coordinate_menu.configure(to = self.array.shape[0] - 1)
        self.on_coordinate_changed()

    def get_currrent_z(self):
        return self.origin[0] + self.coordinate_var.get() * self.spacing[0]

    def update_image(self, layer):
        self.layer = layer

        data = self.array[layer, :, :]

        if self.stretch:
            self.img = Image.fromarray(data).resize((self.canvas_size, self.canvas_size), Image.ANTIALIAS)
            self.stretch_factor = float(data.shape[0]) / self.canvas_size, float(data.shape[1]) / self.canvas_size
        else:
            self.img = Image.fromarray(data)
            self.stretch_factor = 1.0, 1.0

        self.photo = ImageTk.PhotoImage(image = self.img)

        if self.image_on_canvas is None:
            self.image_on_canvas = self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        else:
            self.canvas.itemconfig(self.image_on_canvas, image = self.photo)


def main(args):
    viewer = Array3DViewer(None)

    array = np.random.rand(args.z, args.y, args.x) * 255
    origin = np.asarray((-100., -75., -60.))
    spacing = np.asarray((3.5, 0.5, 0.5))

    viewer.set_array(array, origin, spacing)

    viewer.master.title('Viewer')
    viewer.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Show a random 3d numpy image")
    parser.add_argument("--x", type=int, default = 100, help="define size of x axis")
    parser.add_argument("--y", type=int, default = 100, help="define size of y axis")
    parser.add_argument("--z", type=int, default = 100, help="define size of z axis")
    main(parser.parse_args())
