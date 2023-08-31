import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import SimpleITK as sitk


def plot_data(image, annotation):

    with open(annotation, "r") as file:
        annotation_line = file.read().split(" ")
        annotation_line = [float(y) for y in annotation_line[1:]]

    im = sitk.ReadImage(image)
    size = im.GetSize()
    im_array = sitk.GetArrayFromImage(im)

    try:
        x_centre_norm, y_centre_norm, z_centre_norm, x_size_norm, y_size_norm, z_size_norm = annotation_line
        # need to get start from centre
        x_start_norm = x_centre_norm - x_size_norm / 2
        y_start_norm = y_centre_norm - y_size_norm / 2
        z_start_norm = z_centre_norm - z_size_norm / 2
        x_start, y_start, z_start = [a * b for a, b in zip([x_start_norm, y_start_norm, z_start_norm], size)]
        x_size, y_size, z_size = [a * b for a, b in zip([x_size_norm, y_size_norm, z_size_norm], size)]

        z_slice = int(np.ceil(z_start))
    except ValueError:
        x_start, y_start, z_start = 0, 0, 0
        x_size, y_size, z_size = 0, 0, 0
        z_slice = int(np.ceil(size[2]/2))

    fig, ax = plt.subplots()
    plt.title('YOLO Annotation')
    ax.imshow(np.flipud(im_array[z_slice]), cmap='gray', vmin=0, vmax=300)
    rect = patches.Rectangle((x_start, size[1]-y_start), x_size, y_size, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.show()





