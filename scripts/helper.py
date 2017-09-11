import SimpleITK as sitk
import numpy as np
from PIL import Image
import argparse, cv2, os, csv


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

def load_csv(root, filename):
    annotations = os.path.join(root, "CSVFILES", filename)

    data = {}
    with open(annotations) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row['seriesuid'] in data:
                data[row['seriesuid']] = []
            data[row['seriesuid']].append(row)
    return data

def load_annotations(root):
    return load_csv(root, "annotations.csv")

def load_candidates(root):
    return load_csv(root, "candidates.csv")

def world_to_voxel(coords, origin, spacing):
    return np.absolute(coords - origin) / spacing

def voxel_to_world(coords, origin, spacing):
    return spacing * coords + origin

def normalize_to_grayscale(arr, factor = 255):
    maxHU = 400.
    minHU = -1000.
    data = (arr - minHU) / (maxHU - minHU)
    data[data > 1] = 1.
    data[data < 0] = 0.
    return data * factor

def rescale_patient_images(scan, spacing, target_voxel_mm, is_mask_image=False, verbose=False):
    if verbose:
        print("Spacing: ", spacing)
        print("Shape: ", scan.shape)

    # print "Resizing dim z"
    resize_x = 1.0
    resize_y = float(spacing[0]) / float(target_voxel_mm)
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
    res = cv2.resize(scan, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
    # print "Shape is now : ", res.shape

    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)
    # print "Shape: ", res.shape
    resize_y = float(spacing[1]) / float(target_voxel_mm)
    resize_x = float(spacing[2]) / float(target_voxel_mm)

    # cv2 can handle max 512 channels..
    if res.shape[2] > 512:
        res = res.swapaxes(0, 2)
        res1 = res[:256]
        res2 = res[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = numpy.vstack([res1, res2])
        res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)

    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)

    if verbose:
        print("Shape after: ", res.shape)

    return res

if __name__ == "__main__":
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
