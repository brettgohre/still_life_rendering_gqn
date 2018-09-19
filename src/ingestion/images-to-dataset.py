def _parse_function(filename):
"""
Reads an image from file, decodes it into a dense tensor, and resizes it to a fixed shape
"""
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [IMAGE_W, IMAGE_H])

  return image_resized

import os
file_list = [os.path.join(IMAGE_DIRECTORY, file) for file in os.listdir(IMAGE_DIRECTORY)]
filenames = tf.constant(file_list)

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
 
  
