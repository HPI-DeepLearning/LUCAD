import helper, argparse, os, cv2, math, arrayviewer
import numpy as np


def show_preview(array, origin, spacing):
    viewer = arrayviewer.Array3DViewer(None)
    viewer.set_array(array, origin, spacing)
    viewer.master.title('Preview')
    viewer.mainloop()

def sanitize_coords(coords, min_val):
    result = [int(round(j)) for j in (coords)]
    result = [max(j, min_val) for j in result]
    return result

def assert_debug(boolean, info):
    if boolean:
        return
    print info
    assert False

def export_subset(args, subset, candidates):
    files = os.listdir(os.path.join(args.root, subset))
    files = [i.replace(".mhd","") for i in filter(lambda x: ".mhd" in x, files)]
    files.sort()

    total = sum([len(candidates[f]) for f in files])
    print "Exporting %s with %d candidates..." % (subset, total)
    lb = helper.SimpleLoadingBar("Exporting", total)

    for f in files:

        scan, origin, spacing = helper.load_itk(os.path.join(root, subset, current_file + ".mhd"))

        scan = helper.rescale_patient_images(scan, spacing, target_voxel_mm)
        spacing = np.asarray([target_voxel_mm, target_voxel_mm, target_voxel_mm])
        scan = helper.normalize_to_grayscale(scan).astype(helper.DTYPE)

        total_candidates = len(candidates)

        data = np.zeros((total_candidates, cubesize, cubesize, cubesize), dtype = helper.DTYPE)
        labels = np.zeros((total_candidates), dtype = helper.DTYPE)

        data_filename = os.path.join(subset, "%s_data.npy" % current_file)
        labels_filename = os.path.join(subset, "%s_labels.npy" % current_file)

        cubesize_arr = np.asarray((cubesize, cubesize, cubesize))
        half_cubesize_arr = cubesize_arr / 2

        directory = os.path.join(output, subset)
        if not os.path.exists(directory):
            os.makedirs(directory)

        candidate_num = 0

        with open(os.path.join(output, subset + ".txt"), "a") as handle:
            handle.write(" ".join([data_filename, labels_filename]) + "\n")

        for c in candidates:

            candidate_coords = np.asarray((float(c['coordZ']), float(c['coordY']), float(c['coordX'])))
            voxel_coords = np.round(helper.world_to_voxel(candidate_coords, origin, spacing))

            z0, y0, x0 = sanitize_coords(voxel_coords - half_cubesize_arr, 0)
            z1, y1, x1 = sanitize_coords(voxel_coords + half_cubesize_arr, 0)

            candidate_roi = scan[z0:z1, y0:y1, x0:x1]

            coordinates = [z0, y0, x0, z1, y1, x1]
            assert_debug(min(candidate_roi.shape) > 0, coordinates)
            assert_debug(max(candidate_roi.shape) <= cubesize, coordinates)

            padding = [(int(math.ceil(p / 2.0)), int(p / 2.0)) for p in (cubesize_arr - candidate_roi.shape)]
            padded = np.pad(candidate_roi, padding, 'constant', constant_values = (0,))

            if preview: show_preview(padded, np.asarray((0., 0., 0.)), np.asarray((1., 1., 1.)))

            data[candidate_num,:,:,:] = padded
            labels[candidate_num] = int(c['class'])

            candidate_num += 1
            if loading_bar:
                loading_bar.update_progress(progress + candidate_num)

        np.save(os.path.join(output, data_filename), data)
        np.save(os.path.join(output, labels_filename), labels)

def main(args):
    subsets = ["subset" + str(i) for i in args.subsets]

    candidates = helper.load_candidates(args.root)

    for subset in subsets:
        with open(os.path.join(args.output, subset + ".txt"), "w") as handle:
            handle.write("")

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
