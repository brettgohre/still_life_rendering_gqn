# Converts directory of images to tf.data.Dataset
# Viewpoint information is attached according to sequence of photo in directory.
# Viewpoint format: [x, y, z, sin(yaw), cos(yaw), sin(pitch, cos(pitch)]

import tensorflow as tf
import os
import pickle

path = "."
directory = os.listdir(path)

# step 1: read files and assign (frame, viewpoint) pairs
def _make_dataset():
    new_path = os.path.join(path, 'tf_dataset')
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".jpg")]
    frames = tf.constant(files)
    print(frames)

    # First 130 scenes. 12 facing away from wall.
    views1 = [[-1., -1., 0., 0.707, 0.707, 0., 1.],
        [-0.5, -1., 0., 1., 0., 0., 1.],
        [0.5, -1., 0., 1., 0., 0., 1.],
        [1., -1., 0., 0.707, -0.707, 0., 1.],
        [1., -0.5, 0., 0., -1., 0., 1.],
        [1., 0.5, 0., 0., -1., 0., 1.],
        [1., 1., 0., -0.707, -0.707, 0., 1.],
        [0.5, 1., 0., -1, 0., 0., 1.],
        [-0.5, 1., 0., -1., 0., 0., 1.],
        [-1., 1., 0., -0.707, 0.707, 0., 1.],
        [-1., 0.5, 0., 0., 1., 0., 1.],
        [-1, -0.5, 0., 0., 1., 0., 1.]]

    # Scenes 131 and up. 8 center facing.
    views2 = [[-1., -1., 0., 0.707, 0.707, 0., 1.],
        [0., -1., 0., 1., 0., 0., 1.],
        [1., -1., 0., 0.707, -0.707, 0., 1.],
        [1., 0., 0., 0., -1., 0., 1.],
        [1., 1., 0., -0.707, -0.707, 0., 1.],
        [0., 1., 0., -1., 0., 0., 1.],
        [-1., 1., 0., -0.707, 0.707, 0., 1.],
        [-1., 0., 0., 0., 1., 0., 1.]]

    labels = []

    count = 0
    initial_view_number = 0
    second_view_number = 0

    for file in files:
        if os.path.isfile(os.path.join(path, file)):
            if count < 130:
                labels.append(views1[initial_view_number])
                count += 1
                if (initial_view_number == 0):
                    initial_view_number += 1
                elif (initial_view_number % 11 != 0):
                    initial_view_number += 1
                else:
                    initial_view_number = 0

            else:
                labels.append(views2[second_view_number])
                if (second_view_number == 0):
                    second_view_number += 1
                elif (second_view_number % 7 != 0):
                    second_view_number += 1
                else:
                    second_view_number = 0

    viewpoints = tf.constant(labels)
    print('made?')
    return frames, viewpoints

frames, viewpoints = _make_dataset()

# step 2: create a dataset returning slices of `filenames`
dataset = tf.data.Dataset.from_tensor_slices((frames, viewpoints))

# step 3: parse every image in the dataset using `map`
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image, label

dataset = dataset.map(_parse_function)

location = os.path.join(new_path, os.path.basename(file)

pickle.dump(dataset, open(location,'wb'))
