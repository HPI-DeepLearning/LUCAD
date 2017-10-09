import argparse
import os
import sys
from preparation.candidate_generator import CandidateGenerator
from storage.distributed_storage import DistributedStorage
from storage.candidate_storage import CandidateStorage
from util import helper

import logging


def export_subset(args, subset, candidates):
    files = os.listdir(os.path.join(args.root, subset))
    files = [i.replace(".mhd", "") for i in filter(lambda x: ".mhd" in x, files)]
    files.sort()

    generator = None
    if args.augmentation == "nozflip":
        generator = CandidateGenerator(
            flip = ("", "x", "y", "xy"),
            resize = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
            rotate = "xy",
            translate_limits = (-4, 4),
            factor = args.factor,
            translations = 4
        )
    elif args.augmentation == "xy":
        generator = CandidateGenerator(
            resize = (0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15),
            rotate = "xy",
            translate_limits = (-2, 2),
            translate = "after",
            translate_axes = "xy",
            factor = 10 if args.factor == 0 else args.factor
        )
    elif args.augmentation == "dice":
        generator = CandidateGenerator(
            flip = ("", "x", "y"),
            resize = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
            factor = args.factor,
            rotate = "dice"
        )
    elif args.augmentation == "fonova":
        generator = CandidateGenerator(
            flip = ("", "x", "y"),
            resize = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
            rotate = "xy",
            translate_limits = (-2, 2),
            translate = "after",
            translate_axes = "xy",
            factor = args.factor,
            normalization = "fonova"
        )
    elif args.augmentation == "kok":
        generator = CandidateGenerator(
            flip = ("", "x", "y"),
            rotate = "xy",
            translate_limits = (-1, 1),
            translate = "after",
            translations = 3,
            translate_axes = "xyz",
            factor = args.factor
        )
    elif args.augmentation == "none":
        generator = CandidateGenerator()

    assert generator is not None, "bad augmentation method"
    augment_factor = generator.get_augment_factor()

    original = sum([(len(candidates[f]) if f in candidates else 0) for f in files])

    positive = sum([(sum(1 if i['class'] == '1' else 0 for i in candidates[f]) if f in candidates else 0) for f in files])
    negative = sum([(sum(1 if i['class'] == '0' else 0 for i in candidates[f]) if f in candidates else 0) for f in files])

    positive_augmented = positive * augment_factor
    if args.ratio > -1:
        if positive_augmented == 0 and negative > 0:
            logging.warning("Only negative samples in data set and no positive samples, with ratio %d set!" % args.ratio)
            logging.warning("This means none of the negative samples will be written for %s." % subset)
        negatives_downsampled = min(negative, int(round(positive_augmented * args.ratio)))
    else:
        negatives_downsampled = negative

    total = positive_augmented + negatives_downsampled

    if total == 0:
        return

    logging.info("Augmentation factor for positive samples: %d" % augment_factor)
    logging.info("Positive samples (original/augmented): %d / %d" % (positive, positive_augmented))
    logging.info("Downsampling ratio for negative samples: %d" % args.ratio)
    logging.info("Negative samples (original/downsampled): %d / %d" % (negative, negatives_downsampled))

    root = os.path.join(args.output, subset)
    args_ = [root, total, args.cubesize, negative, negatives_downsampled]
    kwargs = {"shuffle": args.shuffle}
    with DistributedStorage(*args_, **kwargs) if args.storage == "raw" else CandidateStorage(*args_, **kwargs) as storage:
        generator.set_candidate_storage(storage)
        generator.store_info({"augmentation": args.augmentation, "total": total, "original": original, "files": files,
                              "args": args, "positive": positive, "negative": negative, "augmented": positive_augmented,
                              "negatives_downsampled": negatives_downsampled})

        logging.info("Exporting %s..." % subset)
        loading_bar = helper.SimpleLoadingBar("Exporting", total)

        for current_file in files:
            if current_file not in candidates:
                continue

            scan, origin, spacing = helper.load_itk(os.path.join(args.root, subset, current_file + ".mhd"))

            logging.debug("Setting scan of file %s with shape %s, origin %s, spacing %s" % (current_file, scan.shape, origin, spacing))
            generator.set_scan(scan, origin, spacing, args.voxelsize, current_file)

            logging.debug("Generating candidates of file %s" % current_file)
            generator.generate(candidates[current_file], args.cubesize, loading_bar, args.preview)

        generator.store_info({"augmentation": args.augmentation, "total": total, "original": original, "files": files,
                              "args": args, "positive": positive, "negative": negative, "augmented": positive_augmented,
                              "negatives_downsampled": negatives_downsampled}, finished = True)


def main(args):
    subsets = helper.get_filtered_subsets(args.root, args.subsets)

    logging.debug(subsets)

    candidates = helper.load_candidates(args.root, args.test)

    for subset in subsets:
        export_subset(args, subset, candidates)


if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG, stream = sys.stdout)

    parser = argparse.ArgumentParser(description = "prepare dataset for FPRED")
    parser.add_argument("root", type=str, help="containing extracted subset folders and CSVFILES folder")
    parser.add_argument("output", type=str, help="outputfolder, subset folders will be created here")
    parser.add_argument("--storage", type=str, help="raw should be faster", choices = ["memmap", "raw"], default = "memmap")
    parser.add_argument("--augmentation", type=str, help="data augmentation type", choices = ["fonova", "dice", "nozflip", "kok", "xy", "none"], default = "none")
    parser.add_argument("--voxelsize", type=float, help="desired size of voxel in mm for rescaling/normalization", default = 1.0)
    parser.add_argument("--ratio", type=float, help="[N]egatives:[P]ositives ratio (N:P = r:1), negatives will be downsampled", default = -1, metavar="r")
    parser.add_argument("--factor", type=int, help="create a fixed number of random augmentation instead of all", default = 0)
    parser.add_argument("--cubesize", type=int, help="length, height and width of exported cubic sample in voxels", default = 36)
    parser.add_argument("--subsets", type=int, nargs="*", help="the subsets which should be processed", default = (-1,))
    parser.add_argument("--shuffle", action="store_true", help="shuffle while storing the data")
    parser.add_argument("--preview", action="store_true", help="show a preview")
    parser.add_argument("--test", action="store_true", help="test with small candidates csv")
    main(parser.parse_args())
