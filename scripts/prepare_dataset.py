import helper, argparse, os, math, arrayviewer
import numpy as np


def show_preview(array, origin, spacing, name = "Preview"):
    viewer = arrayviewer.Array3DViewer(None)
    viewer.set_array(array, origin, spacing)
    viewer.master.title(name)
    viewer.mainloop()


def sanitize_coords(coordinates, min_val):
    result = [int(round(j)) for j in coordinates]
    result = [max(j, min_val) for j in result]
    return result


def assert_debug(boolean, callback, data):
    if boolean:
        return
    callback(data)
    assert False


class CandidateStorage(object):
    def __init__(self, root, n, cube_size, file_prefix = ""):
        self.root = root
        self.n = n
        self.index = 0
        self.file_prefix = file_prefix if file_prefix == "" else file_prefix + "_"

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        data_filename = os.path.join(self.root, "%sdata.npy" % self.file_prefix)
        self.data_shape = (n, cube_size, cube_size, cube_size)
        labels_filename = os.path.join(self.root, "%slabels.npy" % self.file_prefix)

        self.data = np.memmap(data_filename, dtype = helper.DTYPE, mode = "w+", shape = self.data_shape)
        self.labels = np.memmap(labels_filename, dtype = helper.DTYPE, mode = "w+", shape = n)

    def store_candidate(self, data, label):
        self.data[self.index, :, :, :] = data
        self.labels[self.index] = label
        self.index += 1

    def store_info(self, info_object):
        info_object["written"] = helper.now()
        info_object["revision"] = helper.git_hash()
        info_object["shape"] = self.data_shape
        with open(os.path.join(self.root, "%sinfo.txt" % self.file_prefix), "w") as info_file:
            for k in info_object:
                info_file.write("%s: %s\n" % (k, info_object[k]))


class CandidateGenerator(object):
    def __init__(self, flip = ("",), resize = (1.0,), rotate = "none", augment_class = "1"):
        self.flip = flip
        self.resize = resize
        self.rotate = rotate

        self.scans = []
        self.spacings = []

        self.augment_class = augment_class

        self.started = helper.now()

    def get_augment_factor(self):
        rotation_variants = {"none": 1, "dice": 24}
        return len(self.flip) * len(self.resize) * rotation_variants[self.rotate]

    def set_scan(self, scan, origin, spacing, voxel_size, name = ""):
        self.name = name
        self.original_scan = scan
        self.origin = origin
        self.original_spacing = spacing

        self.create_resized_scans(voxel_size)

    def create_resized_scans(self, base_voxel_size):
        self.scans = []
        self.spacings = []
        for size in self.resize:
            voxel_size = size * base_voxel_size
            rescaled = helper.rescale_patient_images(self.original_scan, self.original_spacing, voxel_size)
            self.spacings.append(np.asarray([voxel_size, voxel_size, voxel_size]))
            self.scans.append(helper.normalize_to_grayscale(rescaled).astype(helper.DTYPE))

    def set_candidate_storage(self, storage):
        self.storage = storage

    def augment_with_rotation(self, data, label, preview):
        # First turn to each side of a 'dice', 0 is original
        #   4
        # 3 0 1 2
        #   5
        for side in range(0, 6):
            rotated = data
            if 0 < side < 4:
                rotated = np.rot90(data, side, axes = (0, 1))
            if side == 4:
                rotated = np.rot90(data, 1, axes = (0, 2))
            if side == 5:
                rotated = np.rot90(data, 1, axes = (2, 0))
            for k in range(0, 4):
                # now rotate the top face: v > ^ <
                final = np.rot90(rotated, k, axes = (1, 2))
                self.store_candidate(final, label, preview)

    def generate_augmented_candidates(self, c, cube_size, cube_size_arr, preview):
        for resize_factor in self.resize:
            data, label = self.generate_single_candidate(c, cube_size, cube_size_arr, resize_factor)
            for flip_axis in self.flip:
                if flip_axis == "":
                    self.augment_with_rotation(data, label, preview)
                elif flip_axis == "x":
                    self.augment_with_rotation(np.flip(data, 2), label, preview)
                elif flip_axis == "y":
                    self.augment_with_rotation(np.flip(data, 1), label, preview)
        return self.get_augment_factor()

    def store_candidate(self, data, label, preview):
        if preview:
            show_preview(data, np.asarray((0., 0., 0.)), np.asarray((1., 1., 1.)))
        self.storage.store_candidate(data, label)

    def generate_candidate(self, c, cube_size, cube_size_arr, preview):
        data, label = self.generate_single_candidate(c, cube_size, cube_size_arr)
        self.store_candidate(data, label, preview)
        return 1

    def generate_single_candidate(self, c, cube_size, cube_size_arr, resize_factor = 1.0):
        index = -1
        for i, f in enumerate(self.resize):
            if abs(f - resize_factor) < 0.01:
                index = i

        candidate_coords = np.asarray((float(c['coordZ']), float(c['coordY']), float(c['coordX'])))
        voxel_coords = np.round(helper.world_to_voxel(candidate_coords, self.origin, self.spacings[index]))

        z0, y0, x0 = sanitize_coords(voxel_coords - (cube_size_arr / 2), 0)
        z1, y1, x1 = sanitize_coords(voxel_coords + (cube_size_arr / 2), 0)

        candidate_roi = self.scans[index][z0:z1, y0:y1, x0:x1]

        info = [index, z0, y0, x0, z1, y1, x1]
        assert_debug(min(candidate_roi.shape) > 0, self.show_debug_info, info)
        assert_debug(max(candidate_roi.shape) <= cube_size, self.show_debug_info, info)

        padding = [(int(math.ceil(p / 2.0)), int(p / 2.0)) for p in (cube_size_arr - candidate_roi.shape)]
        padded = np.pad(candidate_roi, padding, 'constant', constant_values = (0,))

        return padded, int(c['class'])

    def show_debug_info(self, info):
        print self.name
        print info
        show_preview(self.scans[info[0]], self.origin, self.spacings[info[0]], "Error on this scan!")

    def generate(self, candidates, cube_size, loading_bar = None, preview = False):
        cube_size_arr = np.asarray((cube_size, cube_size, cube_size))

        for c in candidates:
            if c['class'] == self.augment_class:
                candidates_generated = self.generate_augmented_candidates(c, cube_size, cube_size_arr, preview)
            else:
                candidates_generated = self.generate_candidate(c, cube_size, cube_size_arr, preview)

            if loading_bar is not None:
                loading_bar.advance_progress(candidates_generated)

    def store_info(self, finished = False):
        info = {"started": self.started, "rotate": self.rotate, "resize": self.resize, "flip": self.flip}
        if finished:
            info["finished"] = helper.now()
        self.storage.store_info(info)


def export_subset(args, subset, candidates):
    files = os.listdir(os.path.join(args.root, subset))
    files = [i.replace(".mhd", "") for i in filter(lambda x: ".mhd" in x, files)]
    files.sort()

    generator = CandidateGenerator(
        flip = ("", "x", "y"),
        resize = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
        rotate = "dice"
    )
    augment_factor = generator.get_augment_factor()

    total = sum([sum(augment_factor if i['class'] == '1' else 1 for i in candidates[f]) for f in files])
    original = sum([len(candidates[f]) for f in files])

    print "Reserving storage..."
    storage = CandidateStorage(os.path.join(args.output, subset), total, args.cubesize)
    generator.set_candidate_storage(storage)
    generator.store_info()

    print "Exporting %s with %d (%.2f%% original) candidates..." % (subset, total, float(original) / total * 100)
    loading_bar = helper.SimpleLoadingBar("Exporting", total)

    for current_file in files:
        scan, origin, spacing = helper.load_itk(os.path.join(args.root, subset, current_file + ".mhd"))

        generator.set_scan(scan, origin, spacing, args.voxelsize, current_file)

        generator.generate(candidates[current_file], args.cubesize, loading_bar, args.preview)

    generator.store_info(True)


def main(args):
    subsets = ["subset" + str(i) for i in args.subsets]

    candidates = helper.load_candidates(args.root)

    # todo: augmentation, flip (3D?), resize ratio 0.8 - 1.15, rotate 90 degree
    for subset in subsets:
        export_subset(args, subset, candidates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "prepare dataset for FPRED")
    parser.add_argument("root", type=str, help="containing extracted subset folders and CSVFILES folder")
    parser.add_argument("output", type=str, help="outputfolder, subset folders will be created here")
    parser.add_argument("--voxelsize", type=float, help="desired size of voxel in mm for rescaling/normalization", default = 1.0)
    parser.add_argument("--cubesize", type=int, help="length, height and width of exported cubic sample in voxels", default = 36)
    parser.add_argument("--subsets", type=int, nargs="*", help="the subsets which should be processed", default = range(0, 10))
    parser.add_argument("--preview", action="store_true", help="show a preview")
    main(parser.parse_args())
