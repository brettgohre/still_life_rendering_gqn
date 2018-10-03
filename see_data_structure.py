# Helps look at dataset structure


import tensorflow as tf
import numpy as np
import data_reader

root_path = '.'
data_reader = DataReader(dataset='rooms_ring_camera', context_size=5, root=root_path)
data = data_reader.read(batch_size=12)

with tf.train.SingularMonitoredSession() as sess:
    d = sess.run(data)
    print(np.shape(d))
