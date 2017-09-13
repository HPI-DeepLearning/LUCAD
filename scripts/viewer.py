import helper, argparse, os
from PIL import Image, ImageTk
from arrayviewer import Array3DViewer
import Tkinter as tk
import numpy as np


class Viewer(tk.Frame):
    def __init__(self, master = None, root = ""):
        tk.Frame.__init__(self, master)
        self.root = root
        self.annotations = helper.load_annotations(self.root)
        self.candidates = helper.load_candidates(self.root)
        self.grid(padx = 2, pady = 2)
        self.create_widgets()

        self.image_on_canvas = None
        self.texts_on_canvas = []
        self.circles_on_canvas = []

    def create_widgets(self):
        self.create_subset_selection()
        self.create_file_selection()
        self.create_options()
        self.create_header()
        self.viewer = Array3DViewer(self, row = 3, column = 0, columnspan = 2)
        self.viewer.connect_on_coordinate_changed(self.on_coordinate_changed)

    def make_frame(self, row = 0, column = 0, rowspan = 1, columnspan = 1):
        frame = tk.Frame(self, borderwidth = 3, relief = "ridge")
        frame.grid(row = row, column = column, rowspan = rowspan, columnspan = columnspan, padx = 2, pady = 2, sticky = tk.E + tk.W + tk.S + tk.N)
        return frame

    def create_subset_selection(self):
        self.subsetFrame = self.make_frame(row = 0, column = 0)
        self.subsets = ["subset" + str(i) for i in range(0,10)]
        self.subset_var = tk.StringVar()
        self.subset_var.set(self.subsets[0])
        self.subset_label = tk.Label(self.subsetFrame, text = "Select subset")
        self.subset_label.pack(side = tk.LEFT, padx = 5)
        self.subset_menu = tk.OptionMenu(self.subsetFrame, self.subset_var, *self.subsets, command = self.on_subset_changed)
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

    def create_options(self):
        self.options_frame = self.make_frame(row = 1, column = 0, columnspan = 2)
        self.normalize_var = tk.IntVar()
        self.normalize_var.set(True)
        self.normalize_label = tk.Label(self.options_frame, text = "Rescale?")
        self.normalize_label.pack(side = tk.LEFT, padx = 5)
        self.normalize_menu = tk.Checkbutton(self.options_frame, variable = self.normalize_var, command=self.on_file_changed)
        self.normalize_menu.pack(side = tk.LEFT, padx = 5)
        self.show_var = tk.IntVar()
        self.show_var.set(True)
        self.show_menu = tk.Checkbutton(self.options_frame, variable = self.show_var, command=self.update_annotation)
        self.show_menu.pack(side = tk.RIGHT, padx = 5)
        self.show_label = tk.Label(self.options_frame, text = "Show candidates?")
        self.show_label.pack(side = tk.RIGHT, padx = 5)

    def create_header(self):
        self.header_frame = self.make_frame(row = 2, column = 0, columnspan = 2)
        self.canvas_label = tk.Text(self.header_frame, width = 68, height = 1, borderwidth = 0)
        self.canvas_label.pack(side = tk.TOP, padx = 5)
        self.canvas_label.configure(state = "disabled")

    def on_subset_changed(self, _):
        self.update_files()
        self.file_menu.configure(to = len(self.files) - 1)
        self.file_var.set(0)
        self.on_file_changed()

    def on_file_changed(self, _ = 0):
        self.scan, self.origin, self.spacing = helper.load_itk(os.path.join(self.root, self.subset_var.get(), self.files[self.file_var.get()]))
        self.scan = helper.normalize_to_grayscale(self.scan)

        if self.normalize_var.get():
            target_voxel_mm = 1.0
            self.scan = helper.rescale_patient_images(self.scan, self.spacing, target_voxel_mm)
            self.spacing = np.asarray([target_voxel_mm, target_voxel_mm, target_voxel_mm])

        self.viewer.set_array(self.scan, self.origin, self.spacing)

        self.update_filename()

    def on_coordinate_changed(self, layer):
        self.layer = layer
        self.update_annotation()

    def update_filename(self):
        self.canvas_label.configure(state = "normal")
        self.canvas_label.delete(1.0, tk.END)
        self.canvas_label.insert(tk.END, self.files[self.file_var.get()])
        self.canvas_label.configure(state = "disabled")

    def world_to_voxel(self, coords, origin = None, spacing = None):
        if origin is None:
            origin = self.origin
        if spacing is None:
            spacing = self.spacing
        return helper.world_to_voxel(coords, origin, spacing)

    def get_canvas(self):
        return self.viewer.canvas

    def voxel_to_world(self, coords):
        return helper.voxel_to_world(coords, self.origin, self.spacing)

    def clear_annotations(self):
        for c in self.circles_on_canvas:
            self.get_canvas().delete(c)
        self.circles_on_canvas = []

        for c in self.texts_on_canvas:
            self.get_canvas().delete(c)
        self.texts_on_canvas = []

    @staticmethod
    def make_bbox(coords, radius):
        return coords[1] - radius, coords[0] - radius, coords[1] + radius, coords[0] + radius

    def make_annotations(self, dataset, diameter, color, pos, anchor = 0):
        z_coords = []

        if self.basename in dataset:
            for data in dataset[self.basename]:
                z_coords.append("%.2f" % float(data['coordZ']))
                coords = (float(data['coordZ']), float(data['coordY']), float(data['coordX']))

                v_coords = self.world_to_voxel(coords)
                z_dist = abs(v_coords[0] - self.layer)
                xy_coords = v_coords[1:] / self.viewer.stretch_factor

                if not diameter:
                    if z_dist <= 3:
                        self.circles_on_canvas.append(self.get_canvas().create_oval(self.make_bbox(xy_coords, 5), outline = color))
                    continue

                diameter_mm = (float(data['diameter_mm']), float(data['diameter_mm']), float(data['diameter_mm']))
                v_radius = self.world_to_voxel(diameter_mm, origin = np.asarray((0, 0, 0))) / 2
                radius = v_radius[1] / self.viewer.stretch_factor[0]

                if z_dist <= v_radius[0]:
                    self.circles_on_canvas.append(self.get_canvas().create_oval(self.make_bbox(xy_coords, radius), outline = color))

        if anchor != 0:
            text = ("Annotation(s) at Z: %s" % ", ".join(z_coords)) if len(z_coords) > 0 else "No annotation in this scan."
            self.texts_on_canvas.append(self.get_canvas().create_text(pos, anchor = tk.SW, text = text, fill = "red"))

    def update_annotation(self):
        self.basename, _ = os.path.splitext(self.files[self.file_var.get()])

        self.clear_annotations()
        if self.show_var.get():
            self.make_annotations(self.candidates, False, "yellow", 0, 0)
        self.make_annotations(self.annotations, True, "red", (10, 502), tk.SW)

    def update_files(self):
        files = os.listdir(os.path.join(self.root, self.subset_var.get()))
        self.files = filter(lambda x: ".mhd" in x, files)
        self.files.sort()

    def update_image(self, layer):
        self.layer = layer

        data = helper.normalize_to_grayscale(self.scan[layer,:,:])

        self.img = Image.fromarray(data)
        self.photo = ImageTk.PhotoImage(image = self.img)

        if self.image_on_canvas is None:
            self.image_on_canvas = self.get_canvas().create_image(0, 0, image = self.photo, anchor = tk.NW)
        else:
            self.get_canvas().itemconfig(self.image_on_canvas, image = self.photo)

    def select_subset(self, subset):
        for i, k in enumerate(self.subsets):
            if subset in k:
                self.subset_var.set(k)
                self.on_subset_changed(0);
                return

    def select_files(self, uid):
        self.file_var.set(self.files.index(uid + ".mhd"))
        self.on_file_changed()


def main(args):
    viewer = Viewer(None, args.root)
    if args.subset is not None:
        viewer.select_subset(args.subset)
    if args.seriesuid is not None:
        viewer.select_files(args.seriesuid)
    viewer.master.title('Viewer')
    viewer.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="containing extracted subset folders and CSVFILES folder")
    parser.add_argument("--subset", type=str, help="open specific subset at start")
    parser.add_argument("--seriesuid", type=str, help="open specific file at start")
    main(parser.parse_args())
