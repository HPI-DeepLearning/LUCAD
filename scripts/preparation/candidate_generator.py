import math

import numpy as np
from viewer.arrayviewer import Array3DViewer

from util import helper


def show_preview(array, origin, spacing, name = "Preview"):
    viewer = Array3DViewer(None)
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
    def __init__(self, flip = ("",), resize = (1.0,), rotate = "none", translations = 1, translate_limits = (0, 0),
                 augment_class = "1", translate = "before", translate_axes = "", factor = 0, normalization = "default"):
        self.flip = flip
        self.resize = resize
        self.rotate = rotate
        self.translations = translations
        self.translate = translate
        self.translate_axes = translate_axes
        self.factor = factor
        self.translate_limits = translate_limits
        self.normalization = normalization

        self.scans = []
        self.spacings = []
        self.identity_resize = -1

        self.name = ""
        self.original_scan = None
        self.origin = None
        self.original_spacing = None
        self.storage = None

        self.augment_class = augment_class

        self.started = helper.now()

        self.__rng = np.random.RandomState(42)

    def get_rotation_variants(self):
        return {"none": 1, "dice": 24, "xy": 4}[self.rotate]

    def get_augment_factor(self):
        if self.factor > 0:
            return self.factor
        return len(self.flip) * len(self.resize) * self.get_rotation_variants() * self.translations

    def set_scan(self, scan, origin, spacing, voxel_size, name = ""):
        self.name = name
        self.original_scan = scan
        self.origin = origin
        self.original_spacing = spacing

        self.create_resized_scans(voxel_size)

    def create_resized_scans(self, base_voxel_size):
        self.scans = []
        self.spacings = []
        for i, size in enumerate(self.resize):
            voxel_size = size * base_voxel_size
            rescaled = helper.rescale_patient_images(self.original_scan, self.original_spacing, voxel_size)
            self.spacings.append(np.asarray([voxel_size, voxel_size, voxel_size]))
            self.scans.append(helper.normalize_to_grayscale(rescaled, type = self.normalization).astype(helper.DTYPE))

            if abs(size - 1.0) < 0.01:
                self.identity_resize = i

    def set_candidate_storage(self, storage):
        self.storage = storage

    def augment_with_rotation(self, data, rotate_index):
        if self.rotate == "none":
            return data
        if self.rotate == "xy":
            # now rotate the top face: v > ^ <
            final = np.rot90(data, rotate_index, axes = (1, 2))
            return final
        if self.rotate == "dice":
            side = rotate_index / 4
            k = rotate_index % 4
            print side, k

            # First turn to each side of a 'dice', 0 is original
            #   4
            # 3 0 1 2
            #   5
            rotated = data
            if 0 < side < 4:
                rotated = np.rot90(data, side, axes = (0, 1))
            if side == 4:
                rotated = np.rot90(data, 1, axes = (0, 2))
            if side == 5:
                rotated = np.rot90(data, 1, axes = (2, 0))

            # now rotate the top face: v > ^ <
            final = np.rot90(rotated, k, axes = (1, 2))
            return final

    def generate_translations(self, num):
        t_list = []
        for i in range(0, num):
            t_list.append(self.__rng.randint(self.translate_limits[0], self.translate_limits[1] + 1, size = (3,)))
            if "z" not in self.translate_axes:
                t_list[-1][0] = 0
            if "y" not in self.translate_axes:
                t_list[-1][1] = 0
            if "x" not in self.translate_axes:
                t_list[-1][2] = 0
        return t_list

    def generate_augmented_candidates(self, c, cube_size, cube_size_arr, preview):
        translation_list = self.generate_translations(max(self.translations, self.factor))
        if self.factor > 0:
            for k in range(0, self.factor):
                t = translation_list[k]
                i = self.__rng.randint(0, len(self.resize))
                f = self.__rng.choice(self.flip)
                r = self.__rng.randint(0, self.get_rotation_variants())
                options = {"resize_index": i, "translation": t, "flip_axis": f, "rotate_index": r}
                data, label = self.generate_single_candidate(c, cube_size, cube_size_arr, **options)
                self.store_candidate(data, label, preview, i)
            return self.get_augment_factor()
        for t in translation_list:
            for i in range(0, len(self.resize)):
                for f in self.flip:
                    for r in range(0, self.get_rotation_variants()):
                        options = {"resize_index": i, "translation": t, "flip_axis": f, "rotate_index": r}
                        data, label = self.generate_single_candidate(c, cube_size, cube_size_arr, **options)
                        self.store_candidate(data, label, preview, i)
        return self.get_augment_factor()

    def store_candidate(self, data, label, preview, resize_index = -1):
        if resize_index == -1:
            resize_index = self.identity_resize
        if preview:
            show_preview(data, np.asarray((0., 0., 0.)), self.spacings[resize_index])
        self.storage.store_candidate(data, label)

    def generate_candidate(self, c, cube_size, cube_size_arr, preview):
        data, label = self.generate_single_candidate(c, cube_size, cube_size_arr)
        self.store_candidate(data, label, preview)
        return 1

    def generate_single_candidate(self, c, cube_size, cube_size_arr, resize_index = -1, translation = None,
                                  flip_axis = "", rotate_index = 0):
        if resize_index == -1:
            resize_index = self.identity_resize

        candidate_coords = np.asarray((float(c['coordZ']), float(c['coordY']), float(c['coordX'])))
        if translation is not None and self.translate == "before":
            candidate_coords += translation
        voxel_coords = np.round(helper.world_to_voxel(candidate_coords, self.origin, self.spacings[resize_index]))
        if translation is not None and self.translate == "after":
            voxel_coords += translation

        z0, y0, x0 = sanitize_coords(voxel_coords - (cube_size_arr / 2), 0)
        z1, y1, x1 = sanitize_coords(voxel_coords + (cube_size_arr / 2), 0)

        candidate_roi = self.scans[resize_index][z0:z1, y0:y1, x0:x1]

        info = [resize_index, z0, y0, x0, z1, y1, x1]
        assert_debug(min(candidate_roi.shape) > 0, self.show_debug_info, info)
        assert_debug(max(candidate_roi.shape) <= cube_size, self.show_debug_info, info)

        padding = [(int(math.ceil(p / 2.0)), int(p / 2.0)) for p in (cube_size_arr - candidate_roi.shape)]
        data = np.pad(candidate_roi, padding, 'constant', constant_values = (0,))

        if flip_axis == "" and rotate_index == 0:
            return data, int(c['class'])

        if flip_axis == "":
            data = self.augment_with_rotation(data, rotate_index)
        elif flip_axis == "x":
            data = self.augment_with_rotation(np.flip(data, 2), rotate_index)
        elif flip_axis == "y":
            data = self.augment_with_rotation(np.flip(data, 1), rotate_index)
        elif flip_axis == "xy":
            data = self.augment_with_rotation(np.flip(np.flip(data, 2), 1), rotate_index)

        return data, int(c['class'])

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
