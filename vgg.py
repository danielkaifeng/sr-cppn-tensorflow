import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib.pyplot as plt
import vggnet
from vggnet import test_init, preprocess_img
from sys import argv

if 0: 
    # More training parameters
    learning_rate = 1e-8
    save_step = max_training_iters
    side_supervision = 3
    display_step = 10
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                 save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:0'):
        checkpoint_path = "models/vgg/OSVOS_parent.ckpt-50000" 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True

        sess = tf.Session(config=config)
        sess, prob, input_image = test_init(checkpoint_path, sess)

img = np.array(Image.open("stone.png"))
image = preprocess_img(img)
res = sess.run(prob, feed_dict={input_image: image})[0,:,:,0]

np.savetxt('feature.txt', res, fmt='%.2f', delimiter=',')

plt.imshow(res)
plt.axis('off')
plt.show()
