import math

import numpy as np

from util import helper


def show_preview(array, origin, spacing, name = "Preview"):
    viewer = scripts.viewer.arrayviewer.Array3DViewer(None)
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


class CandidateGenerator(object):
    def __init__(self, flip = ("",), resize = (1.0,), rotate = "none", translations = 1, translate_limits = (0, 0), augment_class = "1"):
        self.flip = flip
        self.resize = resize
        self.rotate = rotate
        self.translations = translations
        self.translate_limits = translate_limits

        self.scans = []
        self.spacings = []

        self.augment_class = augment_class

        self.started = helper.now()

        self.__rng = np.random.RandomState(42)

    def get_augment_factor(self):
        rotation_variants = {"none": 1, "dice": 24, "xy": 4}
        return len(self.flip) * len(self.resize) * rotation_variants[self.rotate] * self.translations

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
        if self.rotate == "none":
            self.store_candidate(data, label, preview)
        if self.rotate == "xy":
            for k in range(0, 4):
                # now rotate the top face: v > ^ <
                final = np.rot90(data, k, axes = (1, 2))
                self.store_candidate(final, label, preview)
        # First turn to each side of a 'dice', 0 is original
        #   4
        # 3 0 1 2
        #   5
        if self.rotate == "dice":
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
        for i in range(0, self.translations):
            r = self.translate_limits[1] - self.translate_limits[0]
            translation = self.__rng.rand(3) * r + self.translate_limits[0]
            for resize_factor in self.resize:
                data, label = self.generate_single_candidate(c, cube_size, cube_size_arr, resize_factor, translation if self.translations > 1 else None)
                for flip_axis in self.flip:
                    if flip_axis == "":
                        self.augment_with_rotation(data, label, preview)
                    elif flip_axis == "x":
                        self.augment_with_rotation(np.flip(data, 2), label, preview)
                    elif flip_axis == "y":
                        self.augment_with_rotation(np.flip(data, 1), label, preview)
                    elif flip_axis == "xy":
                        self.augment_with_rotation(np.flip(np.flip(data, 2), 1), label, preview)
        return self.get_augment_factor()

    def store_candidate(self, data, label, preview):
        if preview:
            show_preview(data, np.asarray((0., 0., 0.)), np.asarray((1., 1., 1.)))
        self.storage.store_candidate(data, label)

    def generate_candidate(self, c, cube_size, cube_size_arr, preview):
        data, label = self.generate_single_candidate(c, cube_size, cube_size_arr)
        self.store_candidate(data, label, preview)
        return 1

    def generate_single_candidate(self, c, cube_size, cube_size_arr, resize_factor = 1.0, translation = None):
        index = -1
        for i, f in enumerate(self.resize):
            if abs(f - resize_factor) < 0.01:
                index = i

        candidate_coords = np.asarray((float(c['coordZ']), float(c['coordY']), float(c['coordX'])))
        if translation is not None:
            candidate_coords += translation
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

    def store_info(self, info_object = {}, finished = False):
        info_object["started"] = self.started
        info_object["rotate"] = self.rotate
        info_object["resize"] = self.resize
        info_object["flip"] = self.flip
        info_object["translate_limits"] = self.translate_limits
        info_object["translate"] = self.translations
        if finished:
            info_object["finished"] = helper.now()
        self.storage.store_info(info_object)
