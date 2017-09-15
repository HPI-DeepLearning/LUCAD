import argparse
import os
import sys

print sys.path

import viewer.arrayviewer
from preparation.candidate_generator import CandidateGenerator
from storage.distributed_storage import DistributedStorage
from util import helper


def export_subset(args, subset, candidates):
    files = os.listdir(os.path.join(args.root, subset))
    files = [i.replace(".mhd", "") for i in filter(lambda x: ".mhd" in x, files)]
    files.sort()

    generator = CandidateGenerator(
        flip = ("", "x", "y", "xy"),
        resize = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
        rotate = "xy",
        translate_limits = (-1, 1),
        translations = 4
    )
    augment_factor = generator.get_augment_factor()

    total = sum([(sum(augment_factor if i['class'] == '1' else 1 for i in candidates[f]) if f in candidates else 0) for f in files])
    original = sum([(len(candidates[f]) if f in candidates else 0) for f in files])

    if total == 0:
        return

    print "Creating storage..."
    with DistributedStorage(os.path.join(args.output, subset), total, args.cubesize) as storage:
        generator.set_candidate_storage(storage)
        generator.store_info()

        print "Exporting %s with %d (%.2f%% original) candidates..." % (subset, total, float(original) / total * 100)
        loading_bar = helper.SimpleLoadingBar("Exporting", total)

        for current_file in files:
            if current_file not in candidates:
                continue

            scan, origin, spacing = helper.load_itk(os.path.join(args.root, subset, current_file + ".mhd"))

            generator.set_scan(scan, origin, spacing, args.voxelsize, current_file)

            generator.generate(candidates[current_file], args.cubesize, loading_bar, args.preview)

        generator.store_info(True)


def main(args):
    subsets = ["subset" + str(i) for i in args.subsets]

    candidates = helper.load_candidates(args.root, args.test)

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
    parser.add_argument("--test", action="store_true", help="test with small candidates csv")
    main(parser.parse_args())
