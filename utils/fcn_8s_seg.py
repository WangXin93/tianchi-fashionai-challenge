"""
Fully Convolutional Networks (FCNs) for Image Segmentation

Copy from:
http://warmspringwinds.github.io/tensorflow/tf-slim/2017/01/23/fully-convolutional-networks-(fcns)-for-image-segmentation/
"""
import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np

sys.path.append("./tf-image-segmentation/")
sys.path.append("/home/wangx/project/models/research/slim/")

fcn_8s_checkpoint_path = "./fcn_8s_checkpoint/model_fcn8s_final.ckpt"

from tensorflow.contrib import slim

from tf_image_segmentation.models.fcn_8s import FCN_8s
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

#%%
number_of_classes = 21

#image_filename = './0aabd1c980ae90bdcdb71999820327aa.jpg'
#image_filename = './0a2b51a4965598fb1d25171ce61cdbc8.jpg'
#image_filename = './0a6cf698da845a68c7149f06483460fe.jpg'
image_filename = './0a7bece3a5d769bd501661b59ee862ef.jpg'

image_filename_placeholder = tf.placeholder(tf.string)

feed_dict_to_use = {image_filename_placeholder: image_filename}

image_tensor = tf.read_file(image_filename_placeholder)

image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)

image_batch_tensor = tf.expand_dims(image_tensor, axis=0)

# Be careful: after adaptation, network returns final labels
# and not logits
FCN_8s = adapt_network_for_any_size_input(FCN_8s, 32)


pred, fcn_8s_variables_mapping = FCN_8s(image_batch_tensor=image_batch_tensor,
                                        number_of_classes=number_of_classes,
                                        is_training=False)

# The op for initializing the variables
initializer = tf.local_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(initializer)

    saver.restore(sess, fcn_8s_checkpoint_path)

    image_np, pred_np = sess.run([image_tensor, pred], feed_dict=feed_dict_to_use)

    io.imshow(image_np)
    io.show()

    io.imshow(pred_np.squeeze())
    io.show()

#%%
import skimage.morphology

prediction_mask = (pred_np.squeeze() == 15)
# Let's apply some morphological operations to
# create the contour for our sticker
cropped_object = image_np * np.dstack((prediction_mask,) * 3)

square = skimage.morphology.square(5)

temp = skimage.morphology.binary_erosion(prediction_mask, square)

negative_mask = (temp != True)

eroding_countour = negative_mask * prediction_mask

png_transparancy_mask = np.uint8(prediction_mask * 255)

image_shape = cropped_object.shape

png_array = np.zeros(shape=[image_shape[0], image_shape[1], 4], dtype=np.uint8)

png_array[:, :, :3] = cropped_object

png_array[:, :, 3] = png_transparancy_mask

io.imshow(cropped_object)
io.show()
