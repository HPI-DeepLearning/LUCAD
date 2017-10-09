import logging
import argparse
import sys
import os
import numpy as np
from shutil import copyfile
from util import helper


def main(args):
    data_files = []
    label_files = []
    info_files = []
    info_files_dictionary = {}

    subsets = helper.get_filtered_subsets(args.root, args.subsets)

    logging.debug("Subsets for merging: %s" % str(subsets))

    total = 0

    for subset in subsets:
        data_files.append(os.path.join(args.root, subset, "data.npy"))
        label_files.append(os.path.join(args.root, subset, "labels.npy"))
        info_files.append(os.path.join(args.root, subset, "info.txt"))
        info_files_dictionary[subset] = helper.read_info_file(os.path.join(args.root, subset, "info.txt"))
        total += info_files_dictionary[subset]["samples"]

    if not os.path.isdir(os.path.join(args.root, args.output)):
        os.makedirs(os.path.join(args.root, args.output))

    data_path = os.path.join(args.root, args.output, "data.npy")
    label_path = os.path.join(args.root, args.output, "labels.npy")

    merged_info = helper.check_and_combine(info_files_dictionary)
    merged_info["samples"] = str(total)
    new_shape = tuple([total] + merged_info["sample_shape"])
    merged_info["shape"] = str(new_shape)
    logging.debug("Info %s: %s" % (args.root, merged_info))

    logging.debug("New shape: %s" % str(new_shape))

    data_out = np.memmap(data_path, dtype = helper.DTYPE, mode = "w+", shape = new_shape)
    labels_out = np.memmap(label_path, dtype = helper.DTYPE, mode = "w+", shape = new_shape[0])

    index = 0

    lb = helper.SimpleLoadingBar("Merging", new_shape[0])

    for current_file in range(0, len(data_files)):
        info = helper.read_info_file(info_files[current_file])
        shape = info["shape"]

        if len(shape) == 4:
            shape = shape[:1] + [1] + shape[1:]

        data = np.memmap(data_files[current_file], dtype = helper.DTYPE, mode = "r")
        data.shape = shape

        labels = np.memmap(label_files[current_file], dtype = helper.DTYPE, mode = "r")
        labels.shape = (shape[0])

        for i in range(0, shape[0]):
            data_out[index + i] = data[i]
            labels_out[index + i] = labels[i]
            lb.advance_progress(1)
        index += shape[0]

    with open(os.path.join(args.root, args.output, "info.txt"), "w") as info_file:
        with open(os.path.join(args.root, subsets[0], "info.txt"), "r") as info_input:
            for line in info_input:
                if any(line.startswith("%s: " % attrib) for attrib in helper.EXC_ATTRIBUTES):
                    continue
                info_file.write(line)
        for k in ["samples", "shape"]:
            info_file.write("%s: %s\n" % (k, merged_info[k]))


if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG, stream = sys.stdout)

    parser = argparse.ArgumentParser(description="merge subsets of dataset")
    parser.add_argument("root", type=str, help="containing extracted subset folders and CSVFILES folder")
    parser.add_argument("--output", type=str, help="subfolder containing merged result", default="merged")
    parser.add_argument("--subsets", type=int, nargs="*", help="the subsets which should be processed", default = (-1,))

    main(parser.parse_args())
