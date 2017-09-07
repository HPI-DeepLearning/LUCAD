import SimpleITK as sitk
import numpy as np
from PIL import Image
import argparse
'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''

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

def normalize_to_grayscale(arr, factor = 255):
    maxHU = 400.
    minHU = -1000.
    data = (arr - minHU) / (maxHU - minHU)
    data[data > 1] = 1.
    data[data < 0] = 0.
    return data * factor

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
