import yaml
from ultralytics import YOLO
import os
import glob
from sklearn.model_selection import train_test_split
import numpy as np
from check_annotations import plot_data
import random

HOMEDIR = os.path.expanduser('~/')

mediaflux = 'Z:/'

if os.path.exists(HOMEDIR + 'mediaflux'):
    mediaflux = HOMEDIR + 'mediaflux/'

codec = mediaflux + 'CTA/CODEC-IV/'
codec_yolo = mediaflux + 'CTA/CODEC-IV/YOLO/'
all_images = glob.glob(codec + 'CODEC-IV/sub-train*/*MeanArterialPhase.nii.gz')
all_images.sort()
all_annotations = glob.glob(codec_yolo + 'annotations/*')
all_annotations.sort()

testing_images = glob.glob(codec + 'CODEC-IV/sub-test*/*MeanArterialPhase.nii.gz')

len_train = int(np.ceil(0.6 * len(all_images)))
len_validation = len(all_images) - len_train
training_ids, validation_ids = train_test_split(np.arange(1, len(all_images) + 1),
                                                train_size=len_train,
                                                test_size=len_validation,
                                                random_state=42,
                                                shuffle=True)

training_images = [path for path in all_images if any(str(num) in path for num in training_ids)]
training_images.sort()
training_annotations = [path for path in all_annotations if any(str(num) in path for num in training_ids)]
training_annotations.sort()

validation_images = [path for path in all_images if any(str(num) in path for num in validation_ids)]
validation_images.sort()
validation_annotations = [path for path in all_annotations if any(str(num) in path for num in validation_ids)]
validation_annotations.sort()

# test random files
image = random.choice(training_images)
name = os.path.basename(image).split('_')[0]
annotation = [file for file in training_annotations if name in file][0]
plot_data(image, annotation)

# TODO: Write yaml file
