import SimpleITK as sitk
import numpy as np
from PIL import Image
import argparse
import cv2
import os
import csv
import sys
import time
import subprocess
import logging


DTYPE = 'u1'
EXC_ATTRIBUTES = ("started", "written", "finished", "shape", "total", "positive", "negative",
                  "original", "augmented", "negatives_downsampled", "samples", "files")


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    except subprocess.CalledProcessError as E:
        if E.returncode == 128:
            return "headless"
        else:
            raise E


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def is_tianchi_dataset(root):
    return os.path.isdir(os.path.join(root, "train_subset00"))


def load_csv(root, filename, constant_columns = ()):
    if is_tianchi_dataset(root):
        annotations = os.path.join(root, "CSVFILES", "train", filename)
    else:
        annotations = os.path.join(root, "CSVFILES", filename)

    data = {}

    if not os.path.isfile(annotations):
        return data

    with open(annotations) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for k, v in constant_columns:
                row[k] = v
            if not row['seriesuid'] in data:
                data[row['seriesuid']] = []
            data[row['seriesuid']].append(row)
    return data


def get_subsets(root):
    subsets = []
    ls = os.listdir(root)
    for i in ls:
        if os.path.isdir(os.path.join(root, i)) and "subset" in i:
            subsets.append(i)
    return subsets


def get_filtered_subsets(root, selection = (-1,)):
    subsets = get_subsets(root)
    assert all([isinstance(s, int) for s in selection]), "selection needs to contain integers only"

    if -1 in selection:
        return subsets

    if not is_tianchi_dataset(root):
        return_value = filter(lambda x: int(x.replace("subset", "")) in selection, subsets)
        if len(return_value) != len(selection):
            logging.warning("Missing subset(s), subsets were %s, but only found %s" % (selection, return_value))
        return return_value
    else:
        subsets.sort()
        filtered = [subsets[s] for s in selection]
        return filtered


def load_annotations(root):
    return load_csv(root, "annotations.csv")


def load_candidates(root, test = False):
    if is_tianchi_dataset(root):
        return load_csv(root, "annotations.csv", constant_columns = [["class", "1"]])
    if test:
        return load_csv(root, "candidates_test.csv")
    return load_csv(root, "candidates_V2.csv")


def world_to_voxel(coords, origin, spacing):
    return np.absolute(coords - origin) / spacing


def voxel_to_world(coords, origin, spacing):
    return spacing * coords + origin


def normalize_to_grayscale(arr, factor = 255, type = "default"):
    if type == "default":
        minHU, maxHU = -1000., 400.
        minTo, maxTo = 0., 1.
    elif type == "fonova":
        minHU, maxHU = -1000., 1000.
        minTo, maxTo = 1., 1.
    arr = arr.astype("f4")
    np.subtract(arr, minHU, out=arr)
    np.divide(arr, maxHU - minHU, out=arr)
    arr[arr > 1] = maxTo
    arr[arr < 0] = minTo
    np.multiply(arr, factor, out=arr)
    return arr.astype(DTYPE)


def check_and_combine(info_files, exclude = EXC_ATTRIBUTES):

    if len(info_files) == 1:
        return info_files[info_files.keys()[0]]
        # d = info_files[info_files.keys()[0]]
        # return {key: d[key] for key in d if key not in exclude}

    comparison = None
    comparison_subset = ""
    for subset, info_data in info_files.items():
        if comparison is None:
            comparison = info_data
            comparison_subset = subset
            continue
        for key, val in comparison.items():
            if key in exclude:
                continue
            if isinstance(val, list):
                ok = all([x == y for x, y in zip(comparison[key], info_data[key])])
            else:
                ok = comparison[key] == info_data[key]
            assert ok, "Error: %s is different for %s/%s: %s/%s" % (key, str(comparison_subset), str(subset), str(val), str(info_data[key]))

    assert comparison is not None, "Could not find any data in the given info files %s." % info_files

    return {key: comparison[key] for key in comparison if key not in exclude}


def read_info_file(filename):
    data = {}

    if not os.path.isfile(filename):
        return data

    with open(filename) as info_file:
        for line in info_file:
            line = line.strip()
            pos = line.find(": ")

            key = line[:pos]
            value = line[pos + 2:]

            if "Namespace" in value:
                value = [x.replace('"', "").replace("'", "") for x in value.replace("Namespace", "")[1:-1].split(", ")]
                new_value = []
                for e in value:
                    if "=" in e:
                        new_value.append(e)
                    else:
                        new_value[-1] += ","+e
                value = {k: v for k, v in [pair.split("=") for pair in new_value]}

            if ", " in value:
                try:
                    value = [int(x) for x in value[1:-1].split(", ")]
                except ValueError:
                    try:
                        value = [float(x) for x in value[1:-1].split(", ")]
                    except ValueError:
                        value = [x.replace('"', "").replace("'", "") for x in value[1:-1].split(", ")]

            if isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    pass

            data[key] = value

    return data


def rescale_patient_images(scan, spacing, target_voxel_mm, is_mask_image=False, verbose=False):
    logging.debug(("Spacing: %s" % spacing))
    logging.debug(("Shape: %s" % str(scan.shape)))

    # logging.debug("Resizing dim z")
    resize_x = 1.0
    resize_y = float(spacing[0]) / float(target_voxel_mm)
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
    res = cv2.resize(scan, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff

    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)
    resize_y = float(spacing[1]) / float(target_voxel_mm)
    resize_x = float(spacing[2]) / float(target_voxel_mm)

    # cv2 can handle max 512 channels..
    # logging.debug(("Shape: %s" % str(res.shape)))
    if res.shape[2] > 512:
        slice_size = 256
        q, r = divmod(res.shape[2], slice_size)
        res = res.swapaxes(0, 2)
        parts = []
        for i in range(0, q):
            parts.append(res[i * slice_size:(i+1) * slice_size])
        if r > 0:
            parts.append(res[-1 * r:])

        # logging.debug(("Parts: %s" % str([p.shape for p in parts])))

        for i in range(0, len(parts)):
            # logging.debug("Parts: %d" % i)
            parts[i] = parts[i].swapaxes(0, 2)
            parts[i] = cv2.resize(parts[i], dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
            if len(parts[i].shape) == 2:
                parts[i] = np.expand_dims(parts[i], 2)
                logging.debug("Partshape changed: %s" % str(parts[i].shape))
            parts[i] = parts[i].swapaxes(0, 2)
        res = np.vstack(parts)
        parts = None
        res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)

    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)

    logging.debug(("Shape after: %s" % str(res.shape)))

    return res


class SimpleLoadingBar(object):
    def __init__(self, name, max_value, width = 40, stream = sys.stderr):
        self.__width = width
        self.__step = 0
        self.__prev_step = -1
        self.__max = max_value
        self.__stream = stream
        self.__name = name

        self.__start = time.time()
        self.__last_update = time.time()

        self.__current_iteration = 0

        self.write_bar(0)

    def format_time(self, n):
        return "%02d:%02d:%02d" % ((n / 60 / 60), (n / 60) % 60, n % 60)

    def write_bar(self, force_redraw = False):
        seconds = int(time.time() - self.__start)
        time_since_last_update = int(time.time() - self.__last_update)
        if self.__step <= self.__prev_step and time_since_last_update < 5:
            return

        if self.__current_iteration == 0:
            time_string = "--:--:--"
        else:
            total = seconds * self.__max / self.__current_iteration
            time_string = self.format_time(total)

        bar = "[%s%s]" % ("-" * self.__step, " " * (self.__width - self.__step))
        line = "%s %s (%s / %s)" % (bar, self.__name, self.format_time(seconds), time_string)

        self.__stream.write("\r" + line)
        self.__stream.flush()

        self.__prev_step = self.__step
        self.__last_update = time.time()

    def advance_progress(self, iterations_passed):
        self.__current_iteration += iterations_passed
        self.__update_progress()

    def update_progress(self, current_iteration):
        self.__current_iteration = current_iteration
        self.__update_progress()

    def __update_progress(self):
        if self.__current_iteration > self.__max:
            self.__max = self.__current_iteration
        while self.__current_iteration > (self.__max * self.__step / self.__width):
            self.__step += 1
        self.write_bar()

    def finish(self):
        self.__stream.write("\n")


if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type = str, help = "containing extracted subset folders and CSVFILES folder")

    args = parser.parse_args()

    scan, origin, spacing = load_itk(args.root + "/subset2/1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405.mhd")

    print "Shape:", scan.shape
    print "Origin:", origin
    print "Spacing:", spacing

    w, h = 512, 512
    data = normalize_to_grayscale(scan[140,:,:])
    img = Image.fromarray(data).convert('L')
    img.save('test.png')
    img.show()
