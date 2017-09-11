import helper, argparse, os, cv2
import numpy as np

def export(root, output, subset, current_file, candidates, target_voxel_mm, cubesize, start_count_at):
    scan, origin, spacing = helper.load_itk(os.path.join(root, subset, current_file + ".mhd"))

    scan = helper.rescale_patient_images(scan, spacing, target_voxel_mm)
    spacing = np.asarray([target_voxel_mm, target_voxel_mm, target_voxel_mm])

    candidate_num = start_count_at
    with open(os.path.join(output, subset + ".txt"), "a") as handle:
        for c in candidates:
            intermediate_dir = "%03d" % (candidate_num / 1000)
            candidate_filename = os.path.join(subset, intermediate_dir, "%s_%05d.jpg" % (current_file, candidate_num))
            handle.write(" ".join([candidate_filename, c['class']]) + "\n")

            data = np.zeros((cubesize, cubesize, cubesize))

            directory = os.path.join(output, subset, intermediate_dir)
            if not os.path.exists(directory):
                os.makedirs(directory)

            cv2.imwrite(os.path.join(output, candidate_filename), data)
            candidate_num += 1

    return candidate_num

def main(args):
    subsets = ["subset" + str(i) for i in range(0,10)]

    candidates = helper.load_candidates(args.root)
    print "Exporting %d candidates..." % sum([len(candidates[c]) for c in candidates])

    total = 0
    for subset in subsets:
        total += len(filter(lambda x: ".mhd" in x, os.listdir(os.path.join(args.root, subset))))
    lb = helper.SimpleLoadingBar("Exporting", total)

    for subset in subsets:
        with open(os.path.join(args.output, subset + ".txt"), "w") as handle:
            handle.write("")

    progress = 0
    num = 0
    for subset in subsets:
        files = os.listdir(os.path.join(args.root, subset))
        files = filter(lambda x: ".mhd" in x, files)
        files.sort()
        for current_file in files:
            current_file = current_file.replace(".mhd","")
            num = export(args.root, args.output, subset, current_file, candidates[current_file], args.voxelsize, args.cubesize, num)
            progress += 1
            lb.update_progress(progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "prepare dataset for FPRED")
    parser.add_argument("root", type=str, help="containing extracted subset folders and CSVFILES folder")
    parser.add_argument("output", type=str, help="outputfolder, subset folders will be created here")
    parser.add_argument("--voxelsize", type=float, help="desired size of voxel in mm for rescasling/normalization", default = 1.0)
    parser.add_argument("--cubesize", type=int, help="length, height and width of exported cubic sample in voxels", default = 36)
    main(parser.parse_args())
