import os
import glob
import SimpleITK as sitk
'''
This code is to create annotated data for the YOLO object detector
I have ground truth data from CODEC-IV in the form of images
but now have to produce a text file for each image with the location of the bounding box
The text file class_id center_x center_y width heightfor each patient should have the information
class_id center_x center_y width height
Where all values are normalised between 0 and 1

I will run YOLO on the original image and on the MIP image
'''

HOMEDIR = os.path.expanduser('~/')

mediaflux = 'Z:/'

if os.path.exists(HOMEDIR + 'mediaflux'):
    mediaflux = HOMEDIR + 'mediaflux/'

codec_annotations = mediaflux + 'CTA/annotation_data/'

annotations_raw = glob.glob(codec_annotations + 'annotations_all/*')
annotations_mipped = glob.glob(codec_annotations + 'annotations_mipped/*')


def get_coords(seg_path):
    im = sitk.ReadImage(seg_path, sitk.sitkUInt8)
    x, y, z = im.GetSize()
    labelfilter = sitk.LabelShapeStatisticsImageFilter()
    labelfilter.Execute(im)
    try:
        centroid = labelfilter.GetCentroid(1)
    except RuntimeError:
        return 0
    xc, yc, zc = im.TransformPhysicalPointToIndex(centroid)
    bbox = labelfilter.GetBoundingBox(1)
    _, _, _, xsize, ysize, zsize = bbox
    xc_norm = xc/x
    yc_norm = yc/y
    zc_norm = zc/z
    xsize_norm = xsize / x
    ysize_norm = ysize / y
    zsize_norm = zsize / z
    return xc_norm, yc_norm, zc_norm, xsize_norm, ysize_norm, zsize_norm


out_dir = codec_annotations + 'annotations_mipped_YOLO/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for annotation in annotations_mipped:
    id = 'INSP_' + os.path.basename(annotation).split('_')[1]
    print(id)
    out_file = os.path.join(out_dir, id + '_yolo.txt')
    try:
        xc_norm, yc_norm, zc_norm, xsize_norm, ysize_norm, zsize_norm = get_coords(annotation)
        with open(out_file, 'w') as myfile:
            myfile.write('1')
            myfile.write(' ')
            myfile.write(str(xc_norm))
            myfile.write(' ')
            myfile.write(str(yc_norm))
            myfile.write(' ')
            myfile.write(str(zc_norm))
            myfile.write(' ')
            myfile.write(str(xsize_norm))
            myfile.write(' ')
            myfile.write(str(ysize_norm))
            myfile.write(' ')
            myfile.write(str(zsize_norm))
    except TypeError:
        # error is thrown due to empty seg, redirect to write an empty file
        with open(out_file, 'w') as myfile:
            myfile.write('')
        continue
