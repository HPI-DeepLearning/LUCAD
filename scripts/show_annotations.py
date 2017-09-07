#!/usr/bin/env python
import load, argparse, os, csv, math
from PIL import Image, ImageTk
import Tkinter as tk
import numpy as np

def load_annotations(root):
    annotations = os.path.join(root, "CSVFILES", "annotations.csv")

    data = {}
    with open(annotations) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            data[row['seriesuid']] = row
    return data

def show_file(root, subset, seriesuid, annotation_data = {}):
    scan, origin, spacing = load.load_itk(os.path.join(root, subset, seriesuid + ".mhd"))
    print scan.shape, origin, spacing

    if annotation_data != {}:
        if seriesuid in annotation_data:
            print annotation_data[seriesuid]
        else:
            print "No annotation!"

def iterate_sets(root, callback):
    subsets = ["subset" + str(i) for i in range(0,10)]

    for subset in subsets:
        files = os.listdir(os.path.join(root, subset))
        for f in files:
            if not os.path.isfile(os.path.join(root, subset, f)):
                continue

            base, extension = os.path.splitext(f)
            if extension == ".raw":
                continue

            callback(root, subset, base)

class Viewer(tk.Frame):
    def __init__(self, master = None, root = ""):
        tk.Frame.__init__(self, master)
        self.root = root
        self.annotations = load_annotations(self.root)
        self.grid(padx = 2, pady = 2)
        self.createWidgets()
        self.bind_all("<Right>", self.on_right_pressed)
        self.bind_all("<Left>", self.on_left_pressed)

    def createWidgets(self):
        self.create_subset_selection()
        self.create_file_selection()
        self.create_coordinates()
        self.create_mouse_field()
        self.create_canvas()

    def make_frame(self, row = 0, column = 0, rowspan = 1, columnspan = 1):
        frame = tk.Frame(self, borderwidth = 3, relief = "ridge")
        frame.grid(row = row, column = column, rowspan = rowspan, columnspan = columnspan, padx = 2, pady = 2, sticky = tk.E + tk.W + tk.S + tk.N)
        return frame

    def create_subset_selection(self):
        self.subsetFrame = self.make_frame(row = 0, column = 0)
        subsets = ["subset" + str(i) for i in range(0,10)]
        self.subset_var = tk.StringVar()
        self.subset_var.set(subsets[0])
        self.subset_label = tk.Label(self.subsetFrame, text = "Select subset")
        self.subset_label.pack(side = tk.LEFT, padx = 5)
        self.subset_menu = tk.OptionMenu(self.subsetFrame, self.subset_var, *subsets, command = self.on_subset_changed)
        self.subset_menu.pack(side = tk.RIGHT, padx = 5, pady = 5)

    def create_file_selection(self):
        self.fileFrame = self.make_frame(row = 0, column = 1)
        self.update_files()
        self.file_var = tk.IntVar()
        self.file_var.set(0)
        self.file_label = tk.Label(self.fileFrame, text = "Select file")
        self.file_label.pack(side = tk.LEFT, padx = 5)
        self.file_menu = tk.Scale(self.fileFrame, variable = self.file_var, to = len(self.files) - 1, showvalue = False, orient=tk.HORIZONTAL, command=self.on_file_changed, length = 200)
        self.file_menu.pack(side = tk.RIGHT, padx = 5)

    def create_coordinates(self):
        self.coordinate_frame = self.make_frame(row = 1, column = 1)

        self.coordinate_label = tk.Label(self.coordinate_frame, text = "")
        self.coordinate_label.pack(side = tk.LEFT, padx = 5)
        self.coordinate_var = tk.IntVar()
        self.coordinate_var.set(0)
        self.coordinate_menu = tk.Scale(self.coordinate_frame, variable = self.coordinate_var, showvalue = False, orient=tk.HORIZONTAL, command=self.on_coordinate_changed, length = 200)
        self.coordinate_menu.pack(side = tk.RIGHT, padx = 5)

    def create_mouse_field(self):
        self.mouse_frame = self.make_frame(row = 1, column = 0)
        self.mouse_label = tk.Label(self.mouse_frame, text = "X: -, Y: -")
        self.mouse_label.pack(side = tk.LEFT, padx = 5)

    def create_canvas(self):
        self.canvas_frame = self.make_frame(row = 2, column = 0, columnspan = 2)
        reject = self.register(lambda: False)
        self.canvas_label = tk.Text(self.canvas_frame, width = 68, height = 1, borderwidth = 0)
        self.canvas_label.pack(side = tk.TOP, padx = 5)
        self.canvas_label.configure(state = "disabled")
        self.canvas = tk.Canvas(self.canvas_frame, width = 512, height = 512)
        self.canvas.pack(side = tk.BOTTOM)
        self.canvas.bind('<Motion>', self.on_mouse_movement)
        self.canvas.bind('<Leave>', self.on_mouse_leave)

        self.image_on_canvas = None
        self.text_on_canvas = None
        self.circle_on_canvas = None

    def on_right_pressed(self, event):
        self.coordinate_var.set(self.coordinate_var.get() + 1)
        self.on_coordinate_changed(0)

    def on_left_pressed(self, event):
        self.coordinate_var.set(self.coordinate_var.get() - 1)
        self.on_coordinate_changed(0)

    def on_mouse_movement(self, event):
        # axis order is z, y, x, and data x and y axis are swapped
        # y = self.voxel_to_world(event.x, 1)
        # x = self.voxel_to_world(event.y, 2)
        # self.mouse_label.configure(text = "X: %.2f, Y: %.2f" % (x, y))
        pass

    def on_mouse_leave(self, event):
        self.mouse_label.configure(text = "X: -, Y: -")

    def on_subset_changed(self, _):
        self.update_files()
        self.file_menu.configure(to = len(self.files) - 1)
        self.file_var.set(0)
        self.on_file_changed(0)

    def on_file_changed(self, _):
        self.scan, self.origin, self.spacing = load.load_itk(os.path.join(self.root, self.subset_var.get(), self.files[self.file_var.get()]))
        print self.scan.shape, self.origin, self.spacing

        self.update_filename()
        self.coordinate_menu.configure(to = self.scan.shape[0] - 1)
        self.coordinate_var.set(0)
        self.on_coordinate_changed(0)

    def on_coordinate_changed(self, _):
        if not hasattr(self, 'scan'):
            return

        self.update_image(self.coordinate_var.get())
        self.coordinate_label.configure(text = "Z: %.2f" % self.get_currrent_z())

        self.update_annotation()

    def update_filename(self):
        self.canvas_label.configure(state = "normal")
        self.canvas_label.delete(1.0, tk.END)
        self.canvas_label.insert(tk.END, self.files[self.file_var.get()])
        self.canvas_label.configure(state = "disabled")

    def get_currrent_z(self):
        return self.origin[0] + self.coordinate_var.get() * self.spacing[0]

    def world_to_voxel(self, coords, spacing = None, origin = None):
        if type(spacing) == type(None):
            spacing = self.spacing
        if type(origin) == type(None):
            origin = self.origin
        return np.absolute(coords - origin) / spacing

    def voxel_to_world(self, coords):
        return self.origin[axis] + value * self.spacing[axis]

    def update_annotation(self):
        basename, _ = os.path.splitext(self.files[self.file_var.get()])

        if basename in self.annotations:
            data = self.annotations[basename]

            coords = (float(data['coordZ']), float(data['coordY']), float(data['coordX']))

            v_coords = self.world_to_voxel(coords)

            diameter_mm = (float(data['diameter_mm']), float(data['diameter_mm']), float(data['diameter_mm']))
            v_radius = self.world_to_voxel(diameter_mm, origin = np.asarray((0, 0, 0))) / 2

            z_dist = abs(v_coords[0] - self.layer)
            radius = 0
            if z_dist <= v_radius[0]:
                radius = v_radius[1]

            if self.circle_on_canvas != None:
                self.canvas.delete(self.circle_on_canvas)
            if radius > 0:
                self.circle_on_canvas = self.canvas.create_oval((v_coords[2] - radius, v_coords[1] - radius, v_coords[2] + radius, v_coords[1] + radius), outline="red")
        else:
            self.canvas.delete(self.circle_on_canvas)

        text = ("Annotation centered at Z: %.2f" % float(data['coordZ'])) if basename in self.annotations else "No annotation in this scan."
        if self.text_on_canvas == None:
            self.text_on_canvas = self.canvas.create_text((10, 502), anchor = tk.SW, text = text, fill = "red")
        else:
            self.canvas.itemconfig(self.text_on_canvas, text = text)

    def update_files(self):
        files = os.listdir(os.path.join(self.root, self.subset_var.get()))
        self.files = filter(lambda x: ".mhd" in x, files)
        self.files.sort()

    def update_image(self, layer):
        self.layer = layer

        data = load.normalize_to_grayscale(self.scan[layer,:,:])

        self.img = Image.fromarray(data)
        self.photo = ImageTk.PhotoImage(image = self.img)

        if self.image_on_canvas == None:
            self.image_on_canvas = self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        else:
            self.canvas.itemconfig(self.image_on_canvas, image = self.photo)

def main(args):
    # annotation_data = load_annotations(args.root)
    # show_file(args.root, "subset0", "1.3.6.1.4.1.14519.5.2.1.6279.6001.979083010707182900091062408058")
    # iterate_sets(args.root, lambda root, subset, seriesuid: show_file(root, subset, seriesuid, annotation_data))
    viewer = Viewer(None, args.root)
    viewer.master.title('Viewer')
    viewer.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="containing extracted subset folders and CSVFILES folder")
    main(parser.parse_args())
