import numpy as np
import tensorflow as tf
from PIL import Image

import argparse
import time
import os
import cPickle

from model import CPPNVAE
from tfrc import get_input_data
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_epochs', type=int, default=100,
                                         help='training epochs')
    parser.add_argument('--display_step', type=int, default=1,
                                         help='display step')
    parser.add_argument('--checkpoint_step', type=int, default=1,
                                         help='checkpoint step')
    parser.add_argument('--batch_size', type=int, default=4,
                                         help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                                         help='learning rate for G and VAE')
    parser.add_argument('--learning_rate_vae', type=float, default=0.001,
                                         help='learning rate for VAE')
    parser.add_argument('--learning_rate_d', type=float, default=0.001,
                                         help='learning rate for D')
    parser.add_argument('--keep_prob', type=float, default=1.00,
                                         help='dropout keep probability')
    parser.add_argument('--scale', type=float, default=1.00,
                                         help='dim scale')
    parser.add_argument('--beta1', type=float, default=0.65,
                                         help='adam momentum param for descriminator')
    parser.add_argument('--use_ckpt', action='store_true', default=False,
                                         help='whether to use the checkpoint')
    args = parser.parse_args()
    return args


s = 128
seq_len = 256 * 256 * 3
args = get_args()

#tf_filepath = "/data/ai-datasets/201-FFHQ/tfrecords/face1024.tfrecords"


def main():
    return gen(args)

def gen(args):
    tf_filepath = "/data/ai-datasets/201-FFHQ/tfrecords/face256.tfrecords"
    img = get_input_data(tf_filepath, seq_len, args.batch_size, 10)

    learning_rate = args.learning_rate
    learning_rate_d = args.learning_rate_d
    learning_rate_vae = args.learning_rate_vae
    batch_size = args.batch_size
    training_epochs = args.training_epochs
    display_step = args.display_step
    checkpoint_step = args.checkpoint_step # save training results every check point step
    beta1 = args.beta1
    keep_prob = args.keep_prob
    dirname = 'save'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    x_dim = s
    y_dim = s
    z_dim = 32
    scale = args.scale
    cppn = CPPNVAE(batch_size=batch_size, 
            learning_rate = learning_rate, 
            learning_rate_d = learning_rate_d, 
            learning_rate_vae = learning_rate_vae, 
            beta1 = beta1, 
            keep_prob = keep_prob,
            x_dim = x_dim,
            y_dim = y_dim,
            c_dim = 3, 
            net_depth_g = 3
            )

    # load previously trained model if appilcable
    ckpt = tf.train.get_checkpoint_state(dirname)
    cppn.load_model(dirname)

    batch_images = cppn.sess.run(img)/255.
    batch_images = np.reshape(batch_images, [batch_size, 256, 256, 3])
    batch_images = np.array([cv2.resize(img, (s, s)) for img in batch_images])

    
    #fake = cppn.generate(x_dim=int(x_dim*scale), y_dim=(y_dim*scale), scale=scale)
    print('scale: ', scale)
    fake = cppn.generate(x_dim=x_dim, y_dim=y_dim, scale=scale)
    im = Image.fromarray(np.uint8(fake[0] * 255))
    outpath = "graph/00_pred.png" 
    im.save(outpath)

if __name__ == '__main__':
    main()
